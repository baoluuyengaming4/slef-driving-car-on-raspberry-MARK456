Python 3.12.10 (tags/v3.12.10:0cc8128, Apr  8 2025, 12:21:36) [MSC v.1943 64 bit (AMD64)] on win32
Enter "help" below or click "Help" above for more information.
import RPi.GPIO as GPIO
from time import sleep

GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)

# Lop dieu khien motor don, su dung PWM de dieu chinh toc do
class Motor:
    def __init__(self, in1, in2):
        self.in1 = in1
        self.in2 = in2
        GPIO.setup(self.in1, GPIO.OUT)
        GPIO.setup(self.in2, GPIO.OUT)

        # Thiet lap PWM voi tan so 1kHz
        self.pwm1 = GPIO.PWM(self.in1, 1000)
        self.pwm2 = GPIO.PWM(self.in2, 1000)

        self.pwm1.start(0)  # Bat dau voi duty cycle 0%
        self.pwm2.start(0)

    def forward(self, speed=100):
        # Quay xuoi: chan 1 HIGH theo duty, chan 2 LOW
        self.pwm1.ChangeDutyCycle(speed)
        self.pwm2.ChangeDutyCycle(0)

    def backward(self, speed=100):
        # Quay nguoc: chan 1 LOW, chan 2 HIGH theo duty
        self.pwm1.ChangeDutyCycle(0)
        self.pwm2.ChangeDutyCycle(speed)

    def stop(self):
        self.pwm1.ChangeDutyCycle(0)
        self.pwm2.ChangeDutyCycle(0)


# Lop dieu khien toan bo xe voi 4 motor
class Car:
    def __init__(self, fl, fr, rl, rr):
        self.fl = fl  # front left
        self.fr = fr  # front right
        self.rl = rl  # rear left
        self.rr = rr  # rear right

...     def moveF(self, t=0, speed=100):
...         self.fl.backward(speed)
...         self.fr.backward(speed)
...         self.rl.backward(speed)
...         self.rr.backward(speed)
...         if t > 0:
...             sleep(t)
... 
...     def moveB(self, t=0, speed=100):
...         self.fl.forward(speed)
...         self.fr.forward(speed)
...         self.rl.forward(speed)
...         self.rr.forward(speed)
...         if t > 0:
...             sleep(t)
... 
...     def turn_left(self, t=0, speed=100):
...         self.fl.backward(speed)
...         self.fr.forward(speed)
...         self.rl.backward(speed)
...         self.rr.forward(speed)
...         if t > 0:
...             sleep(t)
... 
...     def turn_right(self, t=0, speed=100):
...         self.fl.forward(speed)
...         self.fr.backward(speed)
...         self.rl.forward(speed)
...         self.rr.backward(speed)
...         if t > 0:
...             sleep(t)
... 
...     def stop(self, t=0):
...         self.fl.stop()
...         self.fr.stop()
...         self.rl.stop()
...         self.rr.stop()
...         if t > 0:
...             sleep(t)
... 
... 
... # Khoi tao cac motor voi chan GPIO tuong ung
... front_left = Motor(4, 6)
... front_right = Motor(13, 12)
... rear_left = Motor(16, 19)
... rear_right = Motor(20, 21)
... 
... # Tao doi tuong xea
... my_car = Car(front_left, front_right, rear_left, rear_right)
... 
... # Vong lap chinh: xe chay toi lien tuc voi toc do co the thay doi
... try:
...     speed = 60
...     while True:
...         my_car.turn_right(speed=speed)
...         sleep(0.1)
... except KeyboardInterrupt:
...     print("Dung chuong trinh.")
...     my_car.stop()
...     GPIO.cleanup()
