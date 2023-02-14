from mytest import square
    
@pytest.fixtures
def input_value():
    return 4

def test_square_give_correct_value(input_value): 
    subject = square(input_value)

    assert subject==4