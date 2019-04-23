from random import randint
import time


class Car:
    tracks = []

    def __init__(self, i, xi, yi, max_age):
        self.i = i
        self.x = xi
        self.y = yi
        self.last_x = xi
        self.last_y = yi
        self.w = 0
        self.h = 0
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
        self.state = '0'
        self.stateLEFT = '0'
        self.stateRIGHT = '0'
        self.age = 0
        self.max_age = max_age
        self.dir = None
        self.prediction = ""
        self.numframe = 0

    def getRGB(self):  # For the RGB colour
        return (self.R, self.G, self.B)

    def getTracks(self):
        return self.tracks

    def getId(self):  # For the ID
        return self.i

    def getState(self):
        return self.state

    def getDir(self):
        return self.dir

    def getX(self):  # for x coordinate
        return self.x

    def getY(self):  # for y coordinate
        return self.y

    def getlastX(self):  # for x coordinate
        return self.last_x

    def getlastY(self):  # for y coordinate
        return self.last_y

    def getW(self):
        return self.w

    def getH(self):
        return self.h

    def getPrediction(self):
        return self.prediction

    def updateCoords(self, xn, yn):
        self.age = 0
        self.tracks.append([self.x, self.y])
        self.x = xn
        self.y = yn

    def setDone(self):
        self.done = True

    def setPrediction(self, predic):
        self.prediction = predic

    def setW(self, Weight):
        self.w = Weight

    def setH(self, Hight):
        self.h = Hight

    def setlastX(self, LX):
        self.last_x = LX

    def setlastY(self, LY):
        self.last_y = LY

    def timedOut(self):
        return self.done

    def going_RIGHT(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][0] < mid_start:
                    self.stateRIGHT = '1'
                    return False
                elif self.stateRIGHT == '1' and self.tracks[-1][0] > mid_end:
                    self.dir = 'right'
                    self.stateRIGHT = '0'
                    self.state = '1'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def going_LEFT(self, mid_start, mid_end):
        if len(self.tracks) >= 2:
            if self.state == '0':
                if self.tracks[-1][0] > mid_end:
                    self.stateLEFT = '1'
                    return False
                elif self.tracks[-1][0] < mid_start and self.stateLEFT == '1':
                    self.dir = 'left'
                    self.stateLEFT = '0'
                    self.state = '1'
                    return True
                else:
                    return False
            else:
                return False
        else:
            return False

    def age_one(self):
        self.age += 1
        if self.age > self.max_age:
            self.done = True
        return True


# Class2

class MultiCar:
    def __init__(self, cars, xi, yi):
        self.cars = cars
        self.x = xi
        self.y = yi
        self.tracks = []
        self.R = randint(0, 255)
        self.G = randint(0, 255)
        self.B = randint(0, 255)
        self.done = False
