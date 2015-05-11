from neuralnets import neuralnets, layers, nnutils

# Fully connected network learning the mnist database
print("Test: a fully connected network on the mnist database.")
print("Loading the mnist database...")
tr_d, te_d = nnutils.load_final_data()
net = neuralnets.SoftmaxFCNetwork(tr_d[0][0].shape, [100, 10], 10, 50, 0.01, 0)
net.train(tr_d[0], tr_d[1])
res = net.test(te_d[0], te_d[1])
print(str(res) + ' / ' + str(len(te_d[0])))
