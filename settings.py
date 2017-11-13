d_of_word = 200 # 词向量的维度
xs_width = d_of_word*2+100
l1_output_width = 20
l2_output_width = 20
loop_round = 1000

rate = 0.01
training_set_size = 100000
test_set_size = 1000

default_threshold = 0.5
threshold_list= [0.1, 0.15, 0.175, 0.2, 0.3]

model_path = "save_game/record3"