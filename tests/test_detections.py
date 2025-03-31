from PIL.Image import Image

from yolox.models import Yolox, YoloxProcessor


class TestDetections:
    def test_detections(self, test_images: list[Image]) -> None:
        for model_id, expected in DETECTIONS_DATA.items():
            print('----------------------------------------')
            print(f'Testing model: {model_id}')
            model = Yolox.from_pretrained(model_id)
            processor = YoloxProcessor(model_id)
            tensor = processor(test_images)
            output = model(tensor)
            actual = processor.postprocess(test_images, output, threshold = 0.65)
            print(actual)
            for a, e in zip(actual, expected):
                for ba, be in zip(a['bboxes'], e['bboxes']):
                    for x, y in zip(ba, be):
                        assert abs(x - y) < 1e-2
                for sa, se in zip(a['scores'], e['scores']):
                    assert abs(sa - se) < 1e-4
                assert a['labels'] == e['labels']


# These baselines were computed with torchvision-0.17.2 / torch-2.2.2. They will reproduce exactly on those torch
# versions. Other torch versions may produce slightly different results; the test incorporates floating-point
# tolerance for this reason.

DETECTIONS_DATA = {
    'yolox_s': [
        {
            'bboxes': [
                (272.36126708984375, 3.5648040771484375, 640.4871826171875, 223.2653350830078),
                (26.643890380859375, 118.68254089355469, 459.80706787109375, 315.089111328125),
                (259.41485595703125, 152.3223114013672, 295.37054443359375, 230.41783142089844),
            ],
            'scores': [0.9417160943584335, 0.8170979975670818, 0.8095869439224117],
            'labels': [7, 2, 12],
        },
        {
            'bboxes': [
                (306.1462707519531, -0.661773681640625, 628.4893798828125, 243.799072265625),
                (252.9065399169922, 227.52923583984375, 587.0886840820312, 478.6072998046875),
                (10.75189208984375, 194.85276794433594, 608.1945190429688, 474.79730224609375),
            ],
            'scores': [0.9228896223610334, 0.7778753937859051, 0.7014229568595098],
            'labels': [45, 50, 45],
        },
        {
            'bboxes': [
                (151.615478515625, 121.58245849609375, 467.77264404296875, 626.9088134765625),
                (237.13978576660156, 92.33920288085938, 261.89569091796875, 124.12648010253906),
                (39.143211364746094, 253.82379150390625, 256.34735107421875, 325.5648193359375),
                (0.37700843811035156, 193.49444580078125, 49.16973114013672, 512.6834106445312),
            ],
            'scores': [0.9026146627036837, 0.8807594746713896, 0.8411656303091419, 0.7363695173835865],
            'labels': [0, 32, 34, 0],
        },
    ],
    'yolox_m': [
        {
            'bboxes': [
                (258.071044921875, 153.78294372558594, 295.64593505859375, 230.4293975830078),
                (17.788238525390625, 116.96869659423828, 456.936767578125, 311.0472106933594),
                (268.9549865722656, 3.2204132080078125, 639.552978515625, 227.1709747314453),
                (168.06326293945312, 109.93966674804688, 278.714111328125, 140.84068298339844),
            ],
            'scores': [0.9438015360564123, 0.9293274398242914, 0.9244454462236291, 0.6791908177622474],
            'labels': [12, 2, 7, 2],
        },
        {
            'bboxes': [
                (1.85009765625, 190.43377685546875, 628.5598754882812, 474.9149169921875),
                (310.0003662109375, -0.11400604248046875, 629.1807250976562, 241.66717529296875),
                (3.9632415771484375, 13.6973876953125, 431.87286376953125, 356.9766845703125),
            ],
            'scores': [0.9206902923141627, 0.9011028525361553, 0.7673841869262219],
            'labels': [45, 45, 45],
        },
        {
            'bboxes': [
                (151.3728485107422, 123.27009582519531, 471.70428466796875, 627.9938354492188),
                (237.3181915283203, 92.74591064453125, 262.6998291015625, 124.12428283691406),
                (-0.06363105773925781, 199.10879516601562, 42.27330017089844, 515.83203125),
                (43.015625, 255.89129638671875, 248.82440185546875, 327.55230712890625),
            ],
            'scores': [0.9294412996061965, 0.9035300569181288, 0.824572293498143, 0.7718706331891312],
            'labels': [0, 32, 0, 34],
        },
    ],
    'yolox_l': [
        {
            'bboxes': [
                (266.55224609375, -0.20783233642578125, 639.8648681640625, 223.93484497070312),
                (258.4273681640625, 153.58741760253906, 295.0760498046875, 232.3666229248047),
                (2.655975341796875, 118.57349395751953, 459.0986633300781, 311.5780029296875),
                (209.98736572265625, 110.51018524169922, 278.888916015625, 140.30113220214844),
            ],
            'scores': [0.9547764066322912, 0.9331239584181183, 0.9127988854742739, 0.8015047842898539],
            'labels': [7, 12, 2, 2],
        },
        {
            'bboxes': [
                (8.01373291015625, 192.7410888671875, 619.24365234375, 475.85382080078125),
                (310.7430419921875, 0.5427398681640625, 629.7476196289062, 244.99168395996094),
                (1.36309814453125, 14.578079223632812, 433.4184265136719, 370.49749755859375),
                (258.5732421875, 232.23861694335938, 568.9124145507812, 473.8310852050781),
            ],
            'scores': [0.9523050819272143, 0.9383309032892484, 0.9285364752896044, 0.6830796094431122],
            'labels': [45, 45, 45, 50],
        },
        {
            'bboxes': [
                (149.04432678222656, 124.79025268554688, 474.92474365234375, 628.5472412109375),
                (237.06727600097656, 93.3296127319336, 262.45574951171875, 123.57549285888672),
                (34.858360290527344, 254.82289123535156, 247.63385009765625, 332.03851318359375),
                (0.06286239624023438, 180.95333862304688, 43.45553970336914, 514.7296142578125),
            ],
            'scores': [0.9551582076262335, 0.8987103117490705, 0.8561213796568978, 0.8463778698491708],
            'labels': [0, 32, 34, 0],
        },
    ],
    'yolox_x': [
        {
            'bboxes': [
                (2.1231842041015625, 118.08766174316406, 459.02618408203125, 316.09649658203125),
                (258.3791198730469, 154.53729248046875, 295.2406311035156, 230.27911376953125),
                (269.196044921875, 32.23992919921875, 639.77978515625, 224.900634765625),
                (160.98602294921875, 109.97015380859375, 278.2338562011719, 141.14566040039062),
            ],
            'scores': [0.9444768181307417, 0.9315928005754159, 0.9288623060389547, 0.6797450161048033],
            'labels': [2, 12, 7, 2],
        },
        {
            'bboxes': [
                (309.87542724609375, 0.21266937255859375, 629.1585083007812, 238.72073364257812),
                (1.49432373046875, 193.5825653076172, 627.2415161132812, 478.5985107421875),
                (0.7376556396484375, 12.82598876953125, 433.14544677734375, 384.5777282714844),
            ],
            'scores': [0.9775241385345268, 0.9574731783027346, 0.9353671078169157],
            'labels': [45, 45, 45],
        },
        {
            'bboxes': [
                (145.92454528808594, 125.65029907226562, 475.572265625, 629.31103515625),
                (236.71348571777344, 93.03581237792969, 262.2492980957031, 124.24107360839844),
                (36.1240234375, 256.0439758300781, 246.8756103515625, 330.9662170410156),
                (0.28395843505859375, 191.75469970703125, 42.65242004394531, 514.560302734375),
            ],
            'scores': [0.9665702924040502, 0.9140369371556645, 0.8323456956829745, 0.8323347393896938],
            'labels': [0, 32, 34, 0],
        },
    ],
    'yolox_tiny': [
        {
            'bboxes': [
                (266.69769287109375, 10.698054313659668, 641.5142211914062, 226.00567626953125),
                (261.3950500488281, 158.86419677734375, 296.5081481933594, 230.33815002441406),
            ],
            'scores': [0.9390157188593093, 0.8303678785484223],
            'labels': [7, 12],
        },
        {
            'bboxes': [
                (333.66583251953125, -2.375758409500122, 628.8615112304688, 244.36590576171875),
                (7.805551528930664, 191.1417999267578, 626.0366821289062, 478.2618408203125),
                (262.2155456542969, 227.8885955810547, 561.689208984375, 473.8562316894531),
            ],
            'scores': [0.8975276906363803, 0.8038670442974158, 0.7662397829717378],
            'labels': [45, 45, 50],
        },
        {
            'bboxes': [
                (153.10641479492188, 124.39726257324219, 478.94927978515625, 630.9003295898438),
                (236.9661407470703, 91.77533721923828, 262.61749267578125, 125.48970794677734),
                (40.31727981567383, 247.88677978515625, 255.65280151367188, 322.5195007324219),
                (0.6052853465080261, 184.91954040527344, 46.53448486328125, 513.2277221679688),
            ],
            'scores': [0.9387520716587368, 0.874984557602474, 0.7543392793374437, 0.6669870865181053],
            'labels': [0, 32, 34, 0],
        },
    ],
    'yolox_nano': [
        {
            'bboxes': [(286.81097412109375, 1.802098274230957, 640.91162109375, 225.0997772216797)],
            'scores': [0.6667016901953495],
            'labels': [7],
        },
        {
            'bboxes': [(310.9555969238281, 1.4645503759384155, 635.9718627929688, 244.2354278564453)],
            'scores': [0.8468274410732874],
            'labels': [45],
        },
        {
            'bboxes': [
                (160.12002563476562, 101.89051818847656, 476.9284362792969, 626.69970703125),
                (237.13267517089844, 94.30076599121094, 260.6517639160156, 125.4441909790039),
            ],
            'scores': [0.9349752252912431, 0.8075243036662982],
            'labels': [0, 32],
        },
    ],
}
