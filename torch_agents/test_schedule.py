from torch_agents import schedule

from pprint import pprint

def test_schedule():
    # Test Flat number of steps and value
    dyn = schedule.Flat(2, 1)
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    dyn.advance()
    assert(not dyn.done)
    assert(float(dyn) == 1.0)
    dyn.advance()
    assert(dyn.done)
    assert(float(dyn) == 1.0)

    # Test Linear
    dyn = schedule.Linear(11, 0.0, 10.0)
    for i in range(11):
        assert(not dyn.done)
        assert(float(dyn) == i)
        dyn.advance()
    assert(dyn.done)
    dyn.advance()
    assert(float(dyn) == 10)

    # Test Linear final value with rounding error
    dyn = schedule.Linear(4, 0.0, 1.0)
    assert(not dyn.done)
    assert(float(dyn) == 0.0)
    dyn.advance()
    float(dyn) # about 0.333
    dyn.advance()
    float(dyn) # about 0.666
    dyn.advance()
    assert(float(dyn) == 1.0)
    dyn.advance()
    assert(dyn.done)
    assert(float(dyn) == 1.0)

    # Test Linear with repeat
    dyn = schedule.Linear(11, 0.0, 10.0, repeat = 2)
    for loop in range(2):
        for i in range(11):
            assert(not dyn.done)
            assert(float(dyn) == i)
            dyn.advance()
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

    for _ in range(3):
        # Create a new dynamic parameter to follow the schedule
        dyn = schd

        # Verify that it follows the schedule
        for _ in range(schd.repeat):
            for step in range(3):
                assert(not dyn.done)
                assert(float(dyn) == 5.0)
                dyn.advance()
            for step in range(21):
                assert(not dyn.done)
                assert(float(dyn) == (5.0 - step/2))
                dyn.advance()

        # after final update
        assert(dyn.done)
        assert(float(dyn) == -5.0) # the final value remains
    
    print("test_schedule() complete")

if __name__=="__main__":
    test_schedule()
