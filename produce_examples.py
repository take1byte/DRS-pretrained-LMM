# a case of drs-text generation
from example_generator import ExampleGenerator

def main():
    eg = ExampleGenerator()

    print("Generation examples:\n")
    eg.mkGenExample(
        text='The court is adjourned until 3:00 p.m. on March 1st.',
        drs='court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1')

    eg.mkGenExample(
        text='There are millions of stars in the universe.',
        drs='be.v.03 Time +1 Theme +3 Location +4 time.n.08 EQU now quantity.n.01 MOR 2000000 star.n.01 Quantity -1 universe.n.01')

    eg.mkGenExample(
        text='The hunting dogs followed the scent of the fox.',
        drs='hunting_dog.n.01 follow.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now scent.n.02 Creator +1 fox.n.01')

    print("\nParsing examples:\n")
    eg.mkParseExample(
        text='The court is adjourned until 3:00 p.m. on March 1st.',
        gold_drs='court.n.01 time.n.08 EQU now adjourn.v.01 Theme -2 Time -1 Finish +1 time.n.08 ClockTime 15:00 MonthOfYear 3 DayOfMonth 1')

    eg.mkParseExample(
        text='There are millions of stars in the universe.',
        gold_drs='be.v.03 Time +1 Theme +3 Location +4 time.n.08 EQU now quantity.n.01 MOR 2000000 star.n.01 Quantity -1 universe.n.01')

    eg.mkParseExample(
        text='The hunting dogs followed the scent of the fox.',
        gold_drs='hunting_dog.n.01 follow.v.01 Agent -1 Time +1 Theme +2 time.n.08 TPR now scent.n.02 Creator +1 fox.n.01')

if __name__ == "__main__":
    main()