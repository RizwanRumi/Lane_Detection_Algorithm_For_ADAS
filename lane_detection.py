import cv2
import numpy as np
import matplotlib.pyplot as plt

class LaneDetection(object):
    """docstring for LaneDetection"""
    def __init__(self):
        pass

    """docstring for image: bgr to gray conversion"""
    def channel_conversion(self, img):
        conv_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return conv_img

    """docstring for image noise reduction"""
    def feature_extraction(self, img):
        # image blurrring, parameters need to tune
        blur = cv2.GaussianBlur(img, (5,5), 0)

        # edge detection by canny
        canny = cv2.Canny(blur, 50, 150)
        return canny

    """docstring for ROI"""
    def region_of_interest(self, img):

        height = img.shape[0]
        width = img.shape[1]
        #print(width, height)

        region_of_interest_vertices = [
            (0, height), (width / 2, height / 2), (width, height)
        ]

        triangle =  np.array([region_of_interest_vertices], np.int32)

        masked_image = self.mask_image(triangle, img)

        roi = cv2.bitwise_and(img, masked_image)
        return roi

    """docstring for roi masking"""
    def mask_image(self, shape, img):
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, shape, 255)
        return mask

    """docstring for hough transform"""
    def hough_transform(self, img):
        lines = cv2.HoughLinesP(img, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=30)
        return lines

    """docstring for lane detection"""
    def lane_detection(self, img):
        # step 2 channel conversion
        copy_img = np.copy(img)
        gray = self.channel_conversion(copy_img)
        # step 3 noise reduction
        canny = self.feature_extraction(gray)
        # step 4 find ROI
        cropped_image = self.region_of_interest(canny)
        # step 5 apply Hough transform
        lines = self.hough_transform(cropped_image)
        # print(lines)

        line_image = np.zeros_like(img)

        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(line_image, (x1, y1), (x2, y2), (255, 0, 0), 5)

        lane = cv2.addWeighted(img, 0.8, line_image, 1, 1)

        return lane

if __name__ == "__main__":

    lane_detection = LaneDetection()

    cap = cv2.VideoCapture('videos/Lane.mp4')

    while (cap.isOpened()):
        ret, frame = cap.read()

        # step 1 read image
        #img = cv2.imread("images/test_image.jpg")
        img = cv2.resize(frame, (320, 320))

        # step 6 display line
        lane = lane_detection.lane_detection(img)

        # plt.imshow(line_image)
        # plt.show()

        cv2.imshow("lane detection", lane)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


