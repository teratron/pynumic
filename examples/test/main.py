class Meta(type):
    # def __new__(mcs, name_class, base, attrs):
    #     attrs.update({'MAX_COORD': 100, 'MIN_COORD': 0})
    #     return type.__new__(mcs, name_class, base, attrs)

    def __init__(cls, name, base, attrs) -> None:
        super().__init__(name, base, attrs)
        cls.MAX_COORD = 100
        cls.MIN_COORD = 0


class Point(metaclass=Meta):
    def get_coords(self) -> tuple[int, int]:
        print(self)
        return 1, 3


if __name__ == "__main__":
    pt = Point()
    print(pt.MAX_COORD)
    print(pt.get_coords())
