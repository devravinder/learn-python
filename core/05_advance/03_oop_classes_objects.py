class Car:
    def __init__(self, brand):
        self.brand = brand

    # every function in a class should take at least one arg else error
    # interpreter passes the self arg/this
    # error if no arg: takes 0 positional arguments but 1 was given
    def start(self):
        print(f"{self.brand} car is starting...")

    def stop(self):
        print("without self")
        print(f"{self.brand} car is stopping...")

c = Car("Tesla")
c.start()
c.stop()
