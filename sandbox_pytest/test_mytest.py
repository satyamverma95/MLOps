from mytest import square
    
def test_square_give_correct_value(): 
    subject = square(2)

    assert subject==4