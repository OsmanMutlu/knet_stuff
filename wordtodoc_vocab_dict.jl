using ExcelReaders
using DataValues
using Knet, JLD
using TextAnalysis

#We create a dict that contains weights for every word in our vocab. Then we multiply words in doc with these weights and create doc vector (in the feats function)

data = readxlsheet("20180502_TimesOfIndia_RawNewsArticles_AgreedByAll.xlsx", "Sheet1")

struct Document
    word::Array
    wvec::Array
    fvec::Array
    bvec::Array
end

#=
#This module is from https://gist.github.com/ozanarkancan/b55c992c5fa26944142c7fd8fdb2b6ff
module WordVec

using PyCall

global const word2vec = PyCall.pywrap(PyCall.pyimport("gensim.models.keyedvectors"))

type Wvec
	model
end

function Wvec(fname::String; bin=true)
	Wvec(word2vec.KeyedVectors["load_word2vec_format"](fname, binary=bin))
end

function getvec(m::Wvec, word::AbstractString)
	vec = nothing
	try
		vec = m.model["__getitem__"](word)
	catch
		vec = m.model["__getitem__"]("unk")
	end
	return vec
end

export Wvec;
export getvec;

end

using WordVec
wvec1 = Wvec("../word-embedding/GoogleNews-vectors-negative300.bin")
=#

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/preprocess.jl
function old_lstm(weight, bias, hidden, cell, input; mask=nothing)
    gates = weight * vcat(input, hidden) .+ bias
    H = size(hidden, 1)
    forget = sigm.(gates[1:H, :])
    ingate = sigm.(gates[1+H:2H, :])
    outgate = sigm.(gates[1+2H:3H, :])
    change = tanh.(gates[1+3H:4H, :])
    (mask != nothing) && (mask = reshape(mask, 1, length(mask)))

    cell = cell .* forget + ingate .* change
    hidden = outgate .* tanh.(cell)

    if mask != nothing
        hidden = hidden .* mask
        cell = cell .* mask
    end
    return (hidden, cell)
end

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/preprocess.jl
#Changed it a little
function wordlstm(fmodel, bmodel, doc::Document)
    # gpu-cpu conversion
    ftype = typeof(fmodel[1])

    # forward lstm
    fvecs = Array{Any}(length(doc.word))
    wforw, bforw = fmodel[1], fmodel[2]
    hidden = cell = ftype(Array{Float32}(xavier(350,1))) # Hiddens are 350x1
    fvecs[1] = hidden
    for i in 1:length(doc.word)-1
        (hidden, cell) = old_lstm(wforw, bforw, hidden, cell, doc.wvec[i])
        fvecs[i+1] = hidden
    end

    # backward lstm
    bvecs = Array{Any}(length(doc.word))
    wback, bback = bmodel[1], bmodel[2]
    hidden = cell = ftype(Array{Float32}(xavier(350,1)))
    bvecs[end] = hidden
    for i in length(doc.word):-1:2
        (hidden, cell) = old_lstm(wback, bback, hidden, cell, doc.wvec[i])
        bvecs[i-1] = hidden
    end
    return fvecs, bvecs
end

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/preprocess.jl
#Changed it a little

function fillallvecs!(corpus; gpufeats=false)
    fmodel = Array{Any}(2)
    bmodel = Array{Any}(2)
    #Maybe pretrained later
    fmodel[1] = (gpufeats ? KnetArray{Float32}(xavier(1400,650)) : Array{Float32}(xavier(1400,650)))
    fmodel[2] = (gpufeats ? KnetArray{Float32}(xavier(1400,1)) : Array{Float32}(xavier(1400,1)))
    bmodel[1] = (gpufeats ? KnetArray{Float32}(xavier(1400,650)) : Array{Float32}(xavier(1400,650)))
    bmodel[2] = (gpufeats ? KnetArray{Float32}(xavier(1400,1)) : Array{Float32}(xavier(1400,1)))
    println("Getting word embeddings and Generating context...")
    for doc in corpus
        #Create context here
        fvecs, bvecs = wordlstm(fmodel, bmodel, doc)
        f = (gpufeats ? KnetArray{Float32} : Array{Float32})
        map(i->push!(doc.fvec, f(i)), fvecs)
        map(i->push!(doc.bvec, f(i)), bvecs)
    end
    return corpus
end

function createcorpus1(data;gpufeats=false)
    ftype = (gpufeats ? KnetArray{Float32} : Array{Float32})
    df = []
    for i in 2:size(data,1)
        typeof(data[i,4]) != typeof(1.0) && continue # Only deal with rows that have a label
        sd = StringDocument(data[i,3])
        remove_corrupt_utf8!(sd)
        remove_punctuation!(sd)
        words = convert(Array{String}, TextAnalysis.tokens(sd))
        words = words[1:end-5]
        embeds = []
        for w in words
            push!(embeds, ftype(getvec(wvec1, w)))
        end
        #doc = WordTokenizers.split_sentences(data[i,3])
        #doc2 = []
        if data[i,4] == 0.0
            data[i,4] = 1
        else
            data[i,4] = 2
        end
        df = vcat(df,Array([data[i,2] Document(words,embeds,[],[]) data[i,4] 0]))
    end

    return df
end

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/model.jl
oparams{T<:Number}(::KnetArray{T},otype; o...)=otype(;o...)
oparams{T<:Number}(::Array{T},otype; o...)=otype(;o...)
oparams(a::Associative,otype; o...)=Dict(k=>oparams(v,otype;o...) for (k,v) in a)
oparams(a,otype; o...)=map(x->oparams(x,otype;o...), a)

# xavier init taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/model.jl
function initx(d...; ftype=Float32, gpufeats=false)
    if gpufeats
        KnetArray{ftype}(xavier(d...))
    else
        Array{ftype}(xavier(d...))
    end
end

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/model.jl
#Changed it a little
function initmodel(inputdim,hiddens,classnumber;gpufeats=false)
    # initialize MLP
    mlpmodel = Any[]
    mlpdims = (inputdim, hiddens..., classnumber)
    #dims = Any[]
    for i in 2:length(mlpdims)
        push!(mlpmodel, initx(mlpdims[i], mlpdims[i-1], gpufeats=gpufeats)) # w
        push!(mlpmodel, initx(mlpdims[i], 1,gpufeats=gpufeats)) # b
    end
    #push!(mlpmodel, dims)
    return mlpmodel
end

#Taken from https://github.com/kirnap/learnbyfun/blob/master/JULIA/pos-tagger/model.jl
function mlp1(w, input)
    #x = dropout(input, pdrop[1])
    for i in 1:2:length(w)-2
        input = relu.(w[i] * input .+ w[i+1]) # To test in linear models
        #x = w[i] * x .+ w[i+1]
        #x = dropout(x, pdrop[2])
    end
    return w[end-1]*input .+ w[end]
end

function oracleloss(model, ind2, ind3, vocab; lval=[], pdrop=(0.0, 0.0))

    total = feats(vocab,model[2],ind2;training=true)
    scores = mlp1(model[1], total)
    #logprobs = logp(scores)
    cval = nll(scores,[ind3])
    #goldind = df[ind,3]
    #cval = -logprobs[goldind]

    push!(lval, (AutoGrad.getval(cval)))
    return cval
end

oraclegrad = grad(oracleloss)

function oracletrain(model, df, vocab, opts;lval=[], pdrop=nothing)
    lval = []
    for ind in 1:size(df,1)
#        println("I'm here")
#        flush(STDOUT)
#        df[ind,4], model[2], words = feats(vocab,df[ind,2];training=true)
        ograds = oraclegrad(model, df[ind,2], df[ind,3], vocab; lval=lval, pdrop=pdrop)
        update!(model, ograds, opts)
#=
        for i in 1:length(words)
            vocab[words[i]] = model[2][i]
        end
=#
    end
    avgloss = mean(lval)
    return avgloss
end

# Balanced Accuracy
function oracleacc(model, df, vocab; pdrop=(0.0, 0.0))
    ntp = 0 #number of true positives
    ntn = 0 #number of true negatives
    npos = 0 #positives
    nneg = 0 #negatives
    for ind in 1:size(df,1)
        df[ind,4] = feats(vocab,model[2],df[ind,2])
        scores = Array(mlp1(model[1], df[ind,4]))
        (val1,pred) = findmax(scores)
        if df[ind,3] == 1
            nneg += 1
            if pred == 1
                ntn += 1
            end
        else
            npos += 1
            if pred == 2
                ntp += 1
            end
        end
    end
    return ((ntp / npos) + (ntn / nneg)) / 2
end

# Normal Accuracy
#=
function oracleacc(mlpmodel, df; pdrop=(0.0, 0.0))
    ncorr = 0
    for ind in 1:size(df,1)
        scores = Array(mlp1(mlpmodel, df[ind,4]))
        (val1,pred) = findmax(scores)
        if pred == df[ind,3]
            ncorr += 1
        end
    end
    return ncorr / size(df,1)
end
=#

function createvocab(df,wordsize)

    vocab = Dict{String,Any}()
    vocab2 = Dict{String,Any}()
    for i in 2:size(df,1)
        for w in df[i,2].word
            if w in keys(vocab)
                #vocab[w] = Array{Float32}(xavier(1))[1]
                vocab[w] += 1
            else
                vocab[w] = 1
            end
        end
    end

    sorted_vocab = sort(collect(zip(values(vocab),keys(vocab))))
#    sorted_vocab = sorted_vocab[1:wordsize-1]
    vocab2["unk"] = 1
    ind = 2
    for w in sorted_vocab
        vocab2[w[2]] = ind
        ind += 1
    end
    return vocab2
end

function feats(vocab,wordmodel,doc;training=false)

    total = fill!(similar(KnetArray{Float32}(doc.wvec[1]), 300, 1),0)
    for j in 1:length(doc.word)
        # This is the input that mlp is going to get
        w = doc.word[j]
        if w in keys(vocab)
            wweight = wordmodel[vocab[w]]
        else
            #wweight = wordmodel[1]
            wweight = 1/length(doc.word)
        end
        total = total .+ KnetArray{Float32}(doc.wvec[j]) .* wweight
    end
    return total
end

function createdf(data; gpufeats=false)
    df = createcorpus1(data,gpufeats=gpufeats)
    data = nothing
    wvec1 = nothing


    df[:,2] = fillallvecs!(df[:,2],gpufeats=gpufeats)
    println("Done generating.")

    df = feats(df)
    println("Generated feats.")

    #corpus = nothing
    #df[:,2] = nothing
    knetgc()
    #return df
    return df

end

function main!(train_df,dev_df,vocab,wordsize;epochs=30,gpufeats=false)

    inputdim = 300
    hiddens = [2048]
    classnumber = 2

    #model = Array{Any}(2)
    model = Any[]
    mlpmodel = initmodel(inputdim,hiddens,classnumber,gpufeats=gpufeats)
    push!(model, mlpmodel)
    #model[1] = mlpmodel
    #model[2] = (gpufeats ? KnetArray{Float32}(xavier(wordsize)) : Array{Float32}(xavier(wordsize)))
    push!(model, (gpufeats ? KnetArray{Float32}(xavier(wordsize)) : Array{Float32}(xavier(wordsize))))
    opts = oparams(model, Adam; gclip=5.0)

#=
    #Seperate corpus and dev
    df = df[shuffle(1:end), :]
    test_df = df[end-199:end,:]
    sep = size(df,1) - round(Int,size(df,1)/5 - 200)
    train_df = df[1:sep,:]
    dev_df = df[sep+1:end-200,:]
=#

    println("Start Training...")
    flush(STDOUT)

    best_acc = 0.0
    acc1 = 0.0
    for i in 1:epochs
        lval = []
        lss = oracletrain(model, train_df, vocab, opts;lval=lval, pdrop=(0.5, 0.8))
        trnacc = oracleacc(model, train_df, vocab; pdrop=(0.0, 0.0))
        acc1 = oracleacc(model, dev_df, vocab; pdrop=(0.0, 0.0))

        if acc1 > best_acc
            JLD.save("experiment_with_vocab_best_model_only_word_embeds.jld", "model", model, "optims", opts)
            best_acc = acc1
            println("Now best model")
        end

        println("Loss val $lss trn acc $trnacc dev acc $acc1 ...")
	flush(STDOUT)
    end
    JLD.save("experiment_with_vocab_only_word_embeds.jld", "model", model, "optims", opts)
    #println("Final!!! Loss val $lss trn acc $trnacc tst acc $acc1 ...")

    test_df = load("testData_20180901_only_word_embeds.jld")["data"]

    if acc1 < best_acc
        asd = load("experiment_with_vocab_best_model_only_word_embeds.jld")
        model = asd["model"]
        opts = asd["optims"]
        acc2 = oracleacc(model, test_df, vocab)
    else
        acc2 = oracleacc(model, test_df, vocab)
    end

    println("test acc $acc2")
end

#=
alldf = createdf(data,gpufeats=true)
println("Saving data...")
save("timesofindia_with_column2_3.jld", "data", alldf)
=#


println("Loading data...")
flush(STDOUT)
train_df = load("trainData_20180901_only_word_embeds.jld")["data"]
dev_df = load("devData_20180901_only_word_embeds.jld")["data"]
#wordsize = 43426
wordsize = 44151
#vocab = createvocab(train_df,wordsize)
#JLD.save("vocab_latest.jld", "vocab", vocab)
vocab = load("vocab_latest.jld")["vocab"]

Knet.setseed(42)

main!(train_df,dev_df,vocab,wordsize,epochs=30,gpufeats=true)
