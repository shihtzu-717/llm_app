input_text = "나는 최근 파리 여행을 다녀왔다"
input_text_list = input_text.split()
print(input_text_list)

str2idx = {word: idx for idx, word in enumerate(input_text_list)}
idx2str = {idx: word for idx, word in enumerate(input_text_list)}
print(str2idx)
print(idx2str)

input_ids = [str2idx[word] for word in input_text_list]
print(input_ids)

