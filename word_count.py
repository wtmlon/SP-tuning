import pdb
with open("T5_template/QNLI/16-100.txt", 'r') as file:
    dates = file.readlines()
    words = []
    for i in dates:
        words += i.replace("\n", "").split(" ")  # 用空字符来代替换行 words +是为了不被覆盖无+将只有最后一条数据
        setWords = list(set(words))  # 集合自动去重
        num = []  # 统计一个单词出现的次数
    for k in setWords:
        count = 0
        for j in words:
            if k == j:
                count = count + 1
        num.append(count)
    print(num)
    print(setWords)
    pdb.set_trace()
