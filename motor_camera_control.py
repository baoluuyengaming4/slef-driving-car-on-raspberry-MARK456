import cv2
import numpy as np
import picamera
import picamera.array
import RPi.GPIO as GPIO
import time

# Thiết lập GPIO
GPIO.setmode(GPIO.BCM)

# Định nghĩa các chân GPIO cho motor
# Motor 1
MOTOR1_EN = 17  # Enable pin
MOTOR1_IN1 = 27 # Input 1
MOTOR1_IN2 = 22 # Input 2

# Motor 2
MOTOR2_EN = 23
MOTOR2_IN1 = 24
MOTOR2_IN2 = 25

# Motor 3
MOTOR3_EN = 5
MOTOR3_IN1 = 6
MOTOR3_IN2 = 13

# Motor 4
MOTOR4_EN = 19
MOTOR4_IN1 = 26
MOTOR4_IN2 = 21

# Thiết lập các chân GPIO là output
motor_pins = [MOTOR1_EN, MOTOR1_IN1, MOTOR1_IN2,
              MOTOR2_EN, MOTOR2_IN1, MOTOR2_IN2,
              MOTOR3_EN, MOTOR3_IN1, MOTOR3_IN2,
              MOTOR4_EN, MOTOR4_IN1, MOTOR4_IN2]

for pin in motor_pins:
    GPIO.setup(pin, GPIO.OUT)

# Tạo PWM cho các motor
pwm1 = GPIO.PWM(MOTOR1_EN, 100)
pwm2 = GPIO.PWM(MOTOR2_EN, 100)
pwm3 = GPIO.PWM(MOTOR3_EN, 100)
pwm4 = GPIO.PWM(MOTOR4_EN, 100)

# Khởi động PWM
pwm1.start(0)
pwm2.start(0)
pwm3.start(0)
pwm4.start(0)

def control_motor(motor_num, direction, speed):
    """
    Điều khiển motor
    motor_num: 1-4 (số thứ tự motor)
    direction: 'forward' hoặc 'backward'
    speed: 0-100 (tốc độ motor)
    """
    if motor_num == 1:
        pwm = pwm1
        in1 = MOTOR1_IN1
        in2 = MOTOR1_IN2
    elif motor_num == 2:
        pwm = pwm2
        in1 = MOTOR2_IN1
        in2 = MOTOR2_IN2
    elif motor_num == 3:
        pwm = pwm3
        in1 = MOTOR3_IN1
        in2 = MOTOR3_IN2
    elif motor_num == 4:
        pwm = pwm4
        in1 = MOTOR4_IN1
        in2 = MOTOR4_IN2
    else:
        return

    if direction == 'forward':
        GPIO.output(in1, GPIO.HIGH)
        GPIO.output(in2, GPIO.LOW)
    elif direction == 'backward':
        GPIO.output(in1, GPIO.LOW)
        GPIO.output(in2, GPIO.HIGH)
    
    pwm.ChangeDutyCycle(speed)

def stop_motors():
    """Dừng tất cả các motor"""
    pwm1.ChangeDutyCycle(0)
    pwm2.ChangeDutyCycle(0)
    pwm3.ChangeDutyCycle(0)
    pwm4.ChangeDutyCycle(0)

def main():
    try:
        # Khởi tạo camera
        camera = picamera.PiCamera()
        camera.resolution = (640, 480)
        camera.framerate = 30

        # Bắt đầu đọc camera
        with picamera.array.PiRGBArray(camera, size=(640, 480)) as stream:
            for frame in camera.capture_continuous(stream, format="bgr", use_video_port=True):
                image = stream.array

                # Hiển thị hình ảnh
                cv2.imshow("Camera View", image)

                # Xử lý phím điều khiển
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('w'):  # Tiến
                    control_motor(1, 'forward', 50)
                    control_motor(2, 'forward', 50)
                    control_motor(3, 'forward', 50)
                    control_motor(4, 'forward', 50)
                elif key == ord('s'):  # Lùi
                    control_motor(1, 'backward', 50)
                    control_motor(2, 'backward', 50)
                    control_motor(3, 'backward', 50)
                    control_motor(4, 'backward', 50)
                elif key == ord('a'):  # Trái
                    control_motor(1, 'backward', 50)
                    control_motor(2, 'forward', 50)
                    control_motor(3, 'forward', 50)
                    control_motor(4, 'backward', 50)
                elif key == ord('d'):  # Phải
                    control_motor(1, 'forward', 50)
                    control_motor(2, 'backward', 50)
                    control_motor(3, 'backward', 50)
                    control_motor(4, 'forward', 50)
                elif key == ord(' '):  # Dừng
                    stop_motors()

                # Xóa stream cho frame tiếp theo
                stream.truncate(0)

    finally:
        # Dọn dẹp
        stop_motors()
        pwm1.stop()
        pwm2.stop()
        pwm3.stop()
        pwm4.stop()
        GPIO.cleanup()
        cv2.destroyAllWindows()
        camera.close()

if __name__ == "__main__":
    main() 