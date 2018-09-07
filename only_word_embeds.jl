using JLD

mutable struct Document
    word::Array
    wvec::Array
    fvec::Array
    bvec::Array
end

getwordembed(i) = i[351:650]

function asd(df)
    for i in 1:size(df,1)
        doc = df[i,2]
        doc.wvec = map(getwordembed, doc.wvec)
        df[i,2] = doc
    end
    return df
end

df = load("trainData_20180901.jld")["data"]
df = asd(df)
save("trainData_20180901_only_word_embeds.jld", "data", df)

df = load("devData_20180901.jld")["data"]
df = asd(df)
save("devData_20180901_only_word_embeds.jld", "data", df)

df = load("testData_20180901.jld")["data"]
df = asd(df)
save("testData_20180901_only_word_embeds.jld", "data", df)
