import click



# @click.group()
# def test():
#     pass

@click.command()
@click.option('--x', default=4)
def f(x):
    print(x**3)
    return x**2

if __name__ == '__main__':
    f()