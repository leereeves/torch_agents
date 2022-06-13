import dynamic

from pprint import pprint

def test_dynamic():
    # Test Flat number of steps and value
    dyn = dynamic.Flat(2, 1)
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    assert(dyn.done)
    assert(float(dyn) == 1.0)

    # Test Linear
    dyn = dynamic.Linear(11, 0.0, 10.0)
    assert(not dyn.done)
    for i in range(10):
        assert(float(dyn) == i)
        assert(not dyn.done)
    assert(float(dyn) == 10)
    assert(dyn.done)
    assert(float(dyn) == 10)

    # Test sequence
    dyn = dynamic.Sequence([
        dynamic.Flat(3, 5.0),
        dynamic.Linear(21, 5.0, -5.0)
    ])

    for i in range(3):
        assert(float(dyn) == 5.0)
        assert(not dyn.done)
    for i in range(20):
        assert(float(dyn) == (5.0 - i/2))
        assert(not dyn.done)

    # before final update
    assert(not dyn.done)
    assert(float(dyn) == -5.0)
    # after final update
    assert(dyn.done)
    assert(float(dyn) == -5.0) # the final value remains


if __name__=="__main__":
    test_dynamic()
