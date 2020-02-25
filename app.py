from inference import Network
import cv2
import argparse
import sys
import numpy as np

np.set_printoptions(threshold=sys.maxsize)

CPU_EXTENSION = "/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/libcpu_extension_sse4.so"

persons_dict = {}


def get_args():
    # TODO

    parser = argparse.ArgumentParser("Run Inference on input video")

    input_video_desc = "The Location of input file"
    device_desc = "The device name, if not CPU"  # On which device the inference will happen
    model_desc = "The Location of the Model XML"

    parser.add_argument("-input_video", help=input_video_desc, default=0)
    parser.add_argument("-device", help=device_desc, default='CPU')
    parser.add_argument("-model", help=model_desc,
                        default='models/person-detection-retail-0013/FP32/person-detection-retail-0013.xml')

    args = parser.parse_args()

    return args


def findCosineSimilarity(a, b):
    return (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def process_output(frame, result, width, height, reidentification_network, reidentification_net_input_shape,
                   person_dict_count):
    cropped_image = np.array([])
    result = result['detection_out']
    for box in result[0][0]:
        conf = box[2]
        if conf >= 0.6:
            xmin = int(box[3] * width)
            ymin = int(box[4] * height)
            xmax = int(box[5] * width)
            ymax = int(box[6] * height)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cropped_image = frame[ymin:ymax, xmin:xmax]

            if cropped_image.ndim != 1:
                try:
                    # preprocess the cropped image i.e the person image to reidentification model image.
                    process_person_frame = cv2.resize(cropped_image, (
                    reidentification_net_input_shape[3], reidentification_net_input_shape[2]))
                    process_person_frame = process_person_frame.transpose((2, 0, 1))
                    # process_person_frame = process_person_frame.reshape(1, *process_person_frame)
                except Exception as e:
                    print(str(e))
                    continue

                reidentification_net_result = reidentification_network.sync_inference(process_person_frame)

                # print(reidentification_net_result['ip_reid'].shape)

                """if len(arr) == 0:
                    arr.append(reidentification_net_result['ip_reid'][0])
                else:
                    cosine_sim = findCosineSimilarity(arr[0], reidentification_net_result['ip_reid'][0])
                    print(cosine_sim)"""

                if person_dict_count == 0:
                    person_dict_count += 1
                    persons_dict[person_dict_count] = reidentification_net_result['ip_reid'][0]
                    continue

                flag = False
                for person in persons_dict:
                    if findCosineSimilarity(persons_dict[person], reidentification_net_result['ip_reid'][0]) >= 0.5:
                        flag = True
                        break

                if flag == False:
                    person_dict_count += 1
                    persons_dict[person_dict_count] = reidentification_net_result['ip_reid'][0]

    unique_people_text = "Unique People: " + str(person_dict_count)
    cv2.putText(frame, unique_people_text, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), thickness=2)

    return frame, person_dict_count


def infer_on_video(args):
    # TODO

    detection_network = Network()
    detection_network.load_model(model=args.model, device='CPU', cpu_extension=CPU_EXTENSION)

    reidentification_network = Network()
    reidentification_network.load_model(model="models/person-reidentification-retail-0200.xml", device='CPU',
                                        cpu_extension=CPU_EXTENSION)

    capture = cv2.VideoCapture(args.input_video)  # Currently capturing through webcam so 0
    capture.open(args.input_video)

    width = int(capture.get(3))
    height = int(capture.get(4))

    detection_net_input_shape = detection_network.get_input_shape()
    reidentification_net_input_shape = reidentification_network.get_input_shape()

    out = cv2.VideoWriter()

    fourcc = cv2.VideoWriter_fourcc(*'DIVX')

    out = cv2.VideoWriter('out.mp4', fourcc, 30, (width, height))
    persons_dict_count = 0

    while capture.isOpened():

        flag, frame = capture.read()

        if not flag:
            break

        key_pressed = cv2.waitKey(60)

        # Detection frame processed for person detection model
        p_frame = cv2.resize(frame, (detection_net_input_shape[3], detection_net_input_shape[2]))
        p_frame = p_frame.transpose((2, 0, 1))
        p_frame = p_frame.reshape(1, *p_frame.shape)

        # plugin.async_inference(p_frame)
        # result = plugin.extract_output()

        # result retrieved from detection network
        result = detection_network.sync_inference(p_frame)

        out_frame, count = process_output(frame, result, width, height, reidentification_network,
                                          reidentification_net_input_shape, persons_dict_count)

        persons_dict_count = count

        cv2.imshow('Output', out_frame)

        out.write(out_frame)

        if key_pressed == 27:
            break

    out.release()
    capture.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_on_video(args)


main()