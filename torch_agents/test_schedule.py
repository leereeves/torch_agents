from . import schedule

from pprint import pprint

def test_schedule():
    # Test Flat number of steps and value
    dyn = schedule.Flat(2, 1).asfloat()
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    assert(dyn.done)
    assert(float(dyn) == 1.0)

    # Test Linear
    dyn = schedule.Linear(11, 0.0, 10.0).asfloat()
    for i in range(11):
        assert(not dyn.done)
        assert(float(dyn) == i)
    assert(dyn.done)
    assert(float(dyn) == 10)

    # Test Linear final value with rounding error
    dyn = schedule.Linear(4, 0.0, 1.0).asfloat()
    assert(not dyn.done)
    assert(float(dyn) == 0.0)
    float(dyn) # about 0.333
    float(dyn) # about 0.666
    assert(float(dyn) == 1.0)
    assert(dyn.done)
    assert(float(dyn) == 1.0)

    # Test Linear with repeat
    dyn = schedule.Linear(11, 0.0, 10.0, repeat = 2).asfloat()
    for loop in range(2):
        for i in range(11):
            assert(not dyn.done)
            assert(float(dyn) == i)
    assert(dyn.done)
    assert(float(dyn) == 10)

    # Test sequence with reuse and repeat
    schd = schedule.Sequence(
        contents = [
            schedule.Flat(3, 5.0),
            schedule.Linear(21, 5.0, -5.0)
        ], 
        repeat = 5
    )

    for reuse in range(3):
        # Create a new dynamic parameter to follow the schedule
        dyn = schd.asfloat()

        # Verify that it follows the schedule
        for loop in range(schd.repeat):
            for step in range(3):
                assert(not dyn.done)
                assert(float(dyn) == 5.0)
            for step in range(21):
                assert(not dyn.done)
                assert(float(dyn) == (5.0 - step/2))

        # after final update
        assert(dyn.done)
        assert(float(dyn) == -5.0) # the final value remains
    
    print("test_schedule() complete")

if __name__=="__main__":
    test_schedule()
