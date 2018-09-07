using ExcelReaders
using DataValues
using Knet, JLD
using TextAnalysis

mutable struct Document
    word::Array
    wvec::Array
    fvec::Array
    bvec::Array
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
function mlp1(w, input; pdrop=(0.0, 0.0))
#    input = dropout(input, pdrop[1])
    for i in 1:2:length(w)-2
        input = relu.(w[i] * input .+ w[i+1]) # To test in linear models
        #x = w[i] * x .+ w[i+1]
#        input = dropout(input, pdrop[2])
    end
    return w[end-1]*input .+ w[end]
end

function oracleloss(mlpmodel, ind3, ind4; lval=[], pdrop=(0.0, 0.0))

    scores = mlp1(mlpmodel, ind4, pdrop=pdrop)
    #logprobs = logp(scores)
    cval = nll(scores,[ind3])
    #goldind = df[ind,3]
    #cval = -logprobs[goldind]

    push!(lval, (AutoGrad.getval(cval)))
    return cval
end

oraclegrad = grad(oracleloss)

function oracletrain(mlpmodel, df, opts; pdrop=nothing)
    lval = []
    for ind in 1:size(df,1)
        ograds = oraclegrad(mlpmodel, df[ind,3], df[ind,4], lval=lval, pdrop=pdrop)
        update!(mlpmodel, ograds, opts)
    end
    avgloss = mean(lval)
    return avgloss
end

#=
# Balanced Accuracy
function oracleacc(mlpmodel, df; pdrop=(0.0, 0.0))
    ntp = 0 #number of true positives
    ntn = 0 #number of true negatives
    nfp = 0 #number of false positives
    nfn = 0 #number of false negatives
    npos = 0 #positives
    nneg = 0 #negatives
    for ind in 1:size(df,1)
        scores = Array(mlp1(mlpmodel, df[ind,4]))
        (val1,pred) = findmax(scores)
        if df[ind,3] == 1
            nneg += 1
            if pred == 1
                ntn += 1
            else
                nfp += 1
            end
        else
            npos += 1
            if pred == 2
                ntp += 1
            else
                nfn += 1
            end
        end
    end

    return ((ntp / npos) + (ntn / nneg)) / 2
end
=#

#if test f1 else balanced accuracy
function oracleacc(mlpmodel, df; pdrop=(0.0, 0.0), test=false)
    ntp = 0 #number of true positives
    ntn = 0 #number of true negatives
    nfp = 0 #number of false positives
    nfn = 0 #number of false negatives
    npos = 0 #positives
    nneg = 0 #negatives
    for ind in 1:size(df,1)
        scores = Array(mlp1(mlpmodel, df[ind,4]))
        (val1,pred) = findmax(scores)
        if df[ind,3] == 1
            nneg += 1
            if pred == 1
                ntn += 1
            else
                nfp += 1
            end
        else
            npos += 1
            if pred == 2
                ntp += 1
            else
                nfn += 1
            end
        end
    end

    acc2 = ((ntp / npos) + (ntn / nneg)) / 2

    if test
        precision_2 = ntp / (ntp + nfp)
        precision_1 = ntn / (ntn + nfn)
        recall_2 = ntp / (ntp + nfn)
        recall_1 = ntn / (ntn + nfp)
        f1_1 = (2 * precision_1 * recall_1) / (precision_1 + recall_1)
        f1_2 = (2 * precision_2 * recall_2) / (precision_2 + recall_2)
        return acc2, f1_1, f1_2
    end

    return acc2
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

function feats(df)
    for i in 1:size(df,1)
        doc = df[i,2]
        total = fill!(KnetArray{Float32}(300,1),0)
        for j in 1:length(doc.word)
            # This is the input that mlp is going to get
            total = total .+ KnetArray{Float32}(doc.wvec[j] .* (1/length(doc.word)))
        end
        df[i,4] = total
    end
    return df
end

function main!(train_df,dev_df;epochs=30,gpufeats=false)

    inputdim = 300
    hiddens = [2048]
    classnumber = 2

    mlpmodel = initmodel(inputdim,hiddens,classnumber,gpufeats=gpufeats)
    opts = oparams(mlpmodel, Adam; gclip=5.0)

    train_df = feats(train_df)
    dev_df = feats(dev_df)

    println("Start Training...")
    flush(STDOUT)

    best_acc = 0.0
    acc1 = 0.0
    for i in 1:epochs
        lval = []
        lss = oracletrain(mlpmodel, train_df, opts; pdrop=(0.3, 0.5))
        trnacc = oracleacc(mlpmodel, train_df; pdrop=(0.0, 0.0))
        acc1 = oracleacc(mlpmodel, dev_df; pdrop=(0.0, 0.0))

        if acc1 > best_acc
            JLD.save("experiment_first_best_model_only_word_embeds.jld", "model", mlpmodel, "optims", opts)
            best_acc = acc1
            println("Now best model")
        end

        println("Loss val $lss trn acc $trnacc dev acc $acc1 ...")
	flush(STDOUT)
    end
    JLD.save("experiment_first_only_word_embeds.jld", "model", mlpmodel, "optims", opts)
    #println("Final!!! Loss val $lss trn acc $trnacc tst acc $acc1 ...")
    test_df = load("testData_20180901_only_word_embeds.jld")["data"]
    test_df = feats(test_df)

    if acc1 < best_acc
        asd = load("experiment_first_best_model_only_word_embeds.jld")
        mlpmodel = asd["model"]
        opts = asd["optims"]
        acc2, f1_neg, f1_pos = oracleacc(mlpmodel, test_df, test=true)
    else
        acc2, f1_neg, f1_pos = oracleacc(mlpmodel, test_df, test=true)
    end

    println("test acc $acc2")
    println("Negative class $f1_neg, Positive class $f1_pos")
end

Knet.setseed(42)

println("Loading data...")
flush(STDOUT)
train_df = load("trainData_20180901_only_word_embeds.jld")["data"]
dev_df = load("devData_20180901_only_word_embeds.jld")["data"]

main!(train_df,dev_df,epochs=30,gpufeats=true)
