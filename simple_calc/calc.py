from classes import Pair
import click


@click.command()
@click.option('--first-number', prompt='First number', help='First number from pair.', type=float)
@click.option('--second-number', prompt='Second number', help='Second number from pair.', type=float)
@click.option('--operation', prompt='Select operation',
              type=click.Choice(['ADD', 'SUBTRACT', 'MULTIPLY', 'DIVIDE'], case_sensitive=False))
def calc(first_number: float, second_number: float, operation: str):
    """
    This script calculates the basic mathematics operations for the pair of numbers

    Parameters
    ----------
    first_number
        first number from pair
    second_number
        second number form pair
    operation
        type of operation (+,-,*,/)

    Returns
    -------
    Function display a results of calculations
    """
    pair = Pair(num1=first_number, num2=second_number)

    operations = {
        'ADD': pair.add(),
        'SUBTRACT': pair.subtract(),
        'MULTIPLY': pair.multiply(),
        'DIVIDE': pair.divide()
    }
    if operation in operations:
        print(f'{pair.num1} {operation} {pair.num2} the operation of {operation} is equal to: {operations[operation]}')


if __name__ == '__main__':
    calc()
