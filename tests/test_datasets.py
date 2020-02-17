import datasets
from matplotlib import pyplot as plt

def test_pong():
    pong = datasets.get('pong-v1')
    assert pong.num_classes == 8
    train, test = pong.make(400, 100, data_root='../data')
    x, x_t, labels = train[16]
    plt.imshow(x.permute(1, 2, 0).numpy())
    plt.show()