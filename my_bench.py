import torch
import time
def benchmark(model, size):
    elapsed = 0.1
    model = model.to("cpu")
    model.eval()
    num_batches = 1000
    images = torch.randn(size)
    # Run the scripted model on a few batches of images
    for i in range(num_batches):
        start = time.time()
        output = model(images)
        end = time.time()
        elapsed = elapsed + (end-start)*1000

    print('Elapsed time: %3.2f ms' % (elapsed/1000))
    return elapsed
