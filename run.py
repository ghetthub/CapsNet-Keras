import numpy as np
from keras import callbacks

import capsulenet

class args:
    save_dir = "weights/"
    debug = True

    # model
    routings = 1

    # hp
    batch_size = 32
    lr = 0.001
    lr_decay = 1.0
    lam_recon = 0.392

    # training
    epochs = 3
    shift_fraction = 0.1
    digit = 5


(x_train, y_train), (x_test, y_test) = capsulenet.load_mnist()

model, eval_model, manipulate_model = capsulenet.CapsNet(input_shape=x_train.shape[1:],
                                                  n_class=len(np.unique(np.argmax(y_train, 1))),
                                                  routings=args.routings)

capsulenet.train(model=model, data=((x_train, y_train), (x_test, y_test)), args=args)

capsulenet.test(eval_model, data=(x_test, y_test), args=args)

model.save_weights("weights.h5")
