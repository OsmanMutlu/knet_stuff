using ExcelReaders
using DataValues
using JLD, Knet
using TextAnalysis

data = readxlsheet("20180901_TimesOfIndia_RawNewsArticles_AgreedByAll.xlsx", "Sheet1")
data2 = readxlsheet("20180901_theIndianExpress_AgreedByAll.xlsx", "Sheet1")
data3 = readxlsheet("20180801_theHindu_AgreedByAll.xlsx", "Sheet1")
data4 = readxlsheet("20180905_scmp_AgreedByAll.xlsx", "Sheet1")

lmfile = "/scratch/users/omutlu/juliastuff/english_chmodel.jld"

mutable struct Document
    word::Array
    wvec::Array
    fvec::Array
    bvec::Array
end

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
function wordlstm(fmodel, bmodel, doc::Document; gpufeats=false)
    # gpu-cpu conversion
    ftype = (gpufeats ? KnetArray{Float32} : Array{Float32})

    # forward lstm
    fvecs = Array{Any}(length(doc.word))
    wforw, bforw = ftype(fmodel[1]), ftype(fmodel[2])
    hidden = cell = ftype(Array{Float32}(xavier(350,1))) # Hiddens are 300x1
    fvecs[1] = hidden
    for i in 1:length(doc.word)-1
        (hidden, cell) = old_lstm(wforw, bforw, hidden, cell, doc.wvec[i])
        fvecs[i+1] = hidden
    end

    # backward lstm
    bvecs = Array{Any}(length(doc.word))
    wback, bback = ftype(bmodel[1]), ftype(bmodel[2])
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

#Omer's language model instead of randomly generating the weights (https://github.com/kirnap)
#=
    asd = load("english_chmodel.jld")
    fmodel = asd["forw"]
    bmodel = asd["back"]
    asd = nothing
    knetgc()
=#
    println("Getting word embeddings and Generating context...")
    for doc in corpus
        #Create context here
        fvecs, bvecs = wordlstm(fmodel, bmodel, doc, gpufeats=gpufeats)
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
#            push!(embeds, ftype(vcat(getvec(wvec1, w),fill!(Array{Float32}(50,1),0.0))))
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

function feats(df)
    for i in 1:size(df,1)
        doc = df[i,2]
#        total = fill!(similar(doc.fvec[1], 1000, 1),0)
        for j in 1:length(doc.word)
            # This is the input that mlp is going to get
            total = vcat(doc.bvec[j],doc.wvec[j],doc.fvec[j])
            doc.wvec[j] = Array{Float32}(total)
        end
        doc.fvec = Any[]
        doc.bvec = Any[]
        df[i,2] = doc
    end
    return df
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

Knet.setseed(42)
#=
toi_df = createdf(data,gpufeats=true) #Timesofindia
ie_df = createdf(data2,gpufeats=true) #Indianexpress
th_df = createdf(data3,gpufeats=true) #Thehindu
toi_df = vcat(toi_df,ie_df)
toi_df = vcat(toi_df,th_df)
println(size(toi_df,1))

#Seperate train,dev and test
toi_df = toi_df[shuffle(1:end), :]
sep1 = size(toi_df,1) - round(Int,size(toi_df,1)*15/100)
test_df = toi_df[sep1:end,:]
sep2 = sep1 - round(Int,size(toi_df,1)/10)
dev_df = toi_df[sep2:sep1-1,:]
train_df = toi_df[1:sep2-1,:]

println("Saving data...")
save("trainData_20180901.jld", "data", train_df)
save("devData_20180901.jld", "data", dev_df)
save("testData_20180901.jld", "data", test_df)
=#
scmp_df = createdf(data4, gpufeats=true)
save("scmpData_20180905.jld", "data", scmp_df)
