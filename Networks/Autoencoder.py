from __future__ import print_function
import timeit
import Utils.DataHelper as DataHelper
import Utils.CostFHelper as CostFHelper
from pylab import *
from Layers.HiddenLayer import *

# Hyper parameters
DATASET_NAME = '../Dataset/mnist.pkl.gz'
LEARNING_RATE = 0.005
NUM_EPOCH = 500
BATCH_SIZE = 20
PATIENCE = 1000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995
VALIDATION_FREQUENCY = 500
VISUALIZE_FREQUENCY = 5000

''' Load dataset | Create Model | Evaluate Model '''
def autoencoder():
    # Load datasets from local disk or download from the internet
    # We only load the images, not label
    datasets = DataHelper.LoadData(DATASET_NAME)
    trainSetX = datasets[0][0]
    validSetX = datasets[1][0]
    testSetX = datasets[2][0]

    nTrainBatchs = trainSetX.get_value(borrow=True).shape[0]
    nValidBatchs = validSetX.get_value(borrow=True).shape[0]
    nTestBatchs = testSetX.get_value(borrow=True).shape[0]
    nTrainBatchs //= BATCH_SIZE
    nValidBatchs //= BATCH_SIZE
    nTestBatchs //= BATCH_SIZE

    # Create model
    '''
    MODEL ARCHITECTURE

    INPUT    ->    HIDDEN LAYER    ->    OUTPUT
    (28x28)       ( 500 neurons )        (28x28)

    '''
    # Create random state
    rng = numpy.random.RandomState(123)

    # Create shared variable for input
    Index = T.lscalar('Index')
    X = T.matrix('X')
    Y = T.ivector('Y')

    X = X.reshape((BATCH_SIZE, 28 * 28))
    # Hidden layer 0
    hidLayer0 = HiddenLayer(
        rng = rng,
        input = X,
        numIn = 28 * 28,
        numOut = 500,
        activation = T.nnet.sigmoid
    )
    hidLayer0Output = hidLayer0.Output()
    hidLayer0Params = hidLayer0.Params()
    hidLayer0WTranspose = hidLayer0.WTranspose()

    # Hidden layer 1
    hidLayer1 = HiddenLayer(
        rng = rng,
        input = hidLayer0Output,
        numIn = 500,
        numOut = 28 * 28,
        activation=T.tanh,
        W = hidLayer0WTranspose
    )
    hidLayer1Output = hidLayer1.Output()
    hidLayer1Params = hidLayer1.Params()

    # Evaluate model
    costTrain = CostFHelper.MSE(hidLayer1Output, X)

    # List of params from model
    params = hidLayer0Params

    # Define gradient
    grads = T.grad(costTrain, params)

    # Update function
    updates = [
        (param, param - LEARNING_RATE * grad)
        for (param, grad) in zip(params, grads)
    ]

    # Train model
    trainModel = theano.function(
        inputs = [Index],
        outputs = costTrain,
        updates = updates,
        givens = {
            X: trainSetX[Index * BATCH_SIZE : (Index + 1) * BATCH_SIZE]
        }
    )

    # Visualize
    visualizeModel = theano.function(
        inputs=[Index],
        outputs=[hidLayer1Output],
        givens={
            X: trainSetX[Index: Index + 25]
        }

    )

    error = CostFHelper.MSE(hidLayer1Output, X)
    # Valid model
    validModel = theano.function(
        inputs = [Index],
        outputs = error,
        givens = {
            X: validSetX[Index * BATCH_SIZE : (Index + 1) * BATCH_SIZE]
        }
    )

    # Test model
    testModel = theano.function(
        inputs=[Index],
        outputs=error,
        givens={
            X: testSetX[Index * BATCH_SIZE: (Index + 1) * BATCH_SIZE]
        }
    )

    doneLooping = False
    iter = 0
    patience = PATIENCE
    best_error = 1000000
    best_iter = 0
    start_time = timeit.default_timer()
    epoch = 0
    ion()
    while (epoch < NUM_EPOCH) and (not doneLooping):
        epoch = epoch + 1
        for indexBatch in range(nTrainBatchs):
            iter = (epoch - 1) * nTrainBatchs + indexBatch
            cost = trainModel(indexBatch)

            if iter % VALIDATION_FREQUENCY == 0:
                print('Validate model....')
                err = 0;
                for indexValidBatch in range(nValidBatchs):
                    err += validModel(indexValidBatch)
                err /= nValidBatchs
                print('Error = ', err)

                if (err < best_error):
                    if (err < best_error * IMPROVEMENT_THRESHOLD):
                        patience = max(patience, iter * PATIENCE_INCREASE)

                    best_iter = iter
                    best_error = err

                    # Test on test set
                    test_losses = [testModel(i) for i in range(nTestBatchs)]
                    test_score = numpy.mean(test_losses)

        if (patience < iter):
            doneLooping = True
            break


    output = visualizeModel(0)
    figure('Original image')
    for k in range(5):
        for l in range(5):
            temp = trainSetX.get_value(borrow=True)[k * 5 + l].reshape((28, 28))
            subplot(5, 5, k * 5 + l + 1)
            gray()
            axis('off')
            imshow(temp)
            draw()

    figure('Decode image')
    for k in range(5):
        for l in range(5):
            temp = output[0][k * 5 + l].reshape((28, 28))
            subplot(5, 5, k * 5 + l + 1)
            gray()
            axis('off')
            imshow(temp)
            draw()

    end_time = timeit.default_timer()
    print(('Optimization complete. Best validation score of %f %% '
           'obtained at iteration %i, with test performance %f %%') %
          (best_error * 100., best_iter + 1, test_score * 100.))
    print(('The code for file ran for %.2fm' % ((end_time - start_time) / 60.)), file=sys.stderr)
    show()
    return 0


if __name__ == '__main__':
    autoencoder()