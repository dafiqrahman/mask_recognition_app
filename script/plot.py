import cv2


class Annotator:
    def __init__(self, img):
        self.img = img
        self.color_list = [(54, 57, 252), (44, 242, 44), (30, 113, 255)]
        self.lw = max(round(sum(self.img.shape) / 2 * 0.003), 2)
        self.tf = max(self.lw-1, 1)

    def draw_box(self, rect_data):
        # rect_data = result.pandas().xyxy[0].values
        self.box = rect_data[:4]
        self.color = self.color_list[rect_data[5]]
        self.label = rect_data[6]
        p1, p2 = (int(self.box[0]), int(self.box[1])
                  ), (int(self.box[2]), int(self.box[3]))
        cv2.rectangle(self.img, p1, p2, self.color,
                      thickness=self.lw, lineType=cv2.LINE_AA)
        w, h = cv2.getTextSize(
            self.label, 0, fontScale=self.lw / 3, thickness=self.tf)[0]
        outside = p1[1] - h >= 3
        p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
        cv2.rectangle(self.img, p1, p2, self.color, -1, cv2.LINE_AA)  # filled
        cv2.putText(self.img,
                    self.label, (p1[0], p1[1] -
                                 2 if outside else p1[1] + h + 2),
                    0,
                    self.lw / 3,
                    (255, 255, 255),
                    thickness=self.tf,
                    lineType=cv2.LINE_AA)

    def results(self):
        return self.img
