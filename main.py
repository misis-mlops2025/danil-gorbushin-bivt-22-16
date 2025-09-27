def main(*args):
    return sum(args)


if __name__ == '__main__':
    print(main(-1, 2))
    print(main(-1, 2, 3))
    print(main(-1, 2, 3, 4))
    print(main(-1, 2, 3, 5))
