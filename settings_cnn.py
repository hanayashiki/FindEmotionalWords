sequence_length = 20
embedding_size = 200

filter_nums = [150,100,150,150,200,200,200]
filter_sizes = [1,2,3,4,5,6,8]
l1_output_width = 20
l2_output_width = 20

category_count = sequence_length**2

epoch = 1000

training_set_size = 15000
test_set_size = 20000

learning_rate = 0.01

training_source = "emotion_addr.csv"
test_source = "test_sentence.csv"

pair_indices = "pair_indices.csv"

model_path = "record/cnn_record"

batch_size = 32

threshold = 0.7
