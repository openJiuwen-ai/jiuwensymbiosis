"""Published optional policy assets. File names and digests belong to their upstream releases."""

from typing import TypedDict


class ArenaRelease(TypedDict):
    repo: str
    revision: str
    files: dict[str, str]


MICRODUCK_REVISION = "236d1835b03ac7b45d962efbf2fe3a6940206112"
MICRODUCK_FILES = {
    "manifest.json": "5c6a73f95f388f131aaef570f6113f0d36b84a72d5755662cd57e67adf7fd24c",
    "velstand.onnx": "ce894d30b3343036d499bfac1102bdaa58d59a21c1b58401ecb70fb6a2f63e0a",
    "alpha_walking.onnx": "e36332d383997d51401897734cd3e79cf5038406feddb18b4d57ecfb141daa6c",
    "alpha_stand.onnx": "1569268713e40deea795dd2922dba50d3621e15a872855408b6b1b125b1c094b",
    "alpha_sitstand.onnx": "a6f102d9fd655782ba85bfffe18175c34621da719cffcdd673be7982a06fc254",
    "alpha_ground_pick.onnx": "ffbf5109982ff999b0ba53afe86b9ae731bbec679d67fb7f8ab4c52152c88872",
    "ball_kick_left.onnx": "d6928284dccd3dd61e08bf2f760effa74309fbefd97b2b31afb2a60f526d196a",
    "ball_kick_right.onnx": "147a32c388c6b19111b3ac3b550a9a6dc8b8bf267118af4d8c3712522eedb5af",
    "roulade.onnx": "3d60da08fc13f29c1b57f41977aa898132c0d60042100149d8e775affcbca32b",
    "roller.onnx": "7b581097a9bc4782a695fd7654edb0c565649ad593ae439563ddf4b2b2e8a8c7",
    "roller_crouch.onnx": "a1a084be240469c76ac9d3fa44d4792f16d4b1da60398b3ecd3cfc5e2244d990",
}

# Jumper also publishes these two complete exports separately from jumper.app.
JUMPER_EXPORTS = {
    "dance_waist": {
        "actor.onnx": "0a309f2268b381dc0a74467b3728a7f2fd33090f724580945698b35d395fbaa2",
        "layout.json": "a72321d9179a43b08f46fdb6110ef3ad30c3b72b18463a8b53a0ab11a106789d",
        "waist.motion.trajectory.json": "831467b2cc622230cacd0c532254c2b082ef67b43407a714ec284957b5db781b",
    },
    "ref_free_jump": {
        "actor.onnx": "e5e463dc39360de032d695b0f6be6d73dc0ca59b2e9106430df7b04667357e94",
        "layout.json": "7aef8c1300b0ce8b33bdc5f58dd0ea927870aeba65dab49d4de23247ac62fc0c",
    },
}


def jumper_export_files():
    for mode, files in JUMPER_EXPORTS.items():
        for filename, digest in files.items():
            yield mode, filename, f"tasks/jumper/{mode}/out/example/{filename}", digest


# Published separately from the factory set, with the same 61-to-14 ONNX interface.
MICRODUCK_ARENA: dict[str, ArenaRelease] = {
    "sprint_2m": {
        "repo": "pollen-robotics/microduck-sprint-2m-reference",
        "revision": "62afcc461f1d4a5ac22bfbff55ce4900c1afba32",
        "files": {
            "policy.onnx": "adbc1594ccfcafa93ecb3872ab985498207b73e4445a36c5b4f58d9d8701b924",
            "manifest.json": "e29481d3cfea399e4bb568dc87c75087fead638a33df888004a60f59a0d3a0a7",
        },
    },
    "hill_climb_1m5": {
        "repo": "pollen-robotics/microduck-hill-climb-1m5-reference",
        "revision": "d319208d92896353f581be754f6325d01f5e3732",
        "files": {
            "policy.onnx": "ea590cadd5c675bc77558e57d17e43ab1b9f084449dd0e811c33077c01f38110",
            "manifest.json": "3249f4e44ccc00488d9dd5941d97ae543a96d17d450f5bec67beb35f5cb4eeaa",
        },
    },
    "roller_sprint_2m": {
        "repo": "pollen-robotics/microduck-roller-sprint-2m-reference",
        "revision": "c72fb92594e8b415487e87dfeed509141afc5e25",
        "files": {
            "policy.onnx": "01edf63a529a8396d46dac62bb427fda1ce67685611e72ddf790bf46f610c82a",
            "manifest.json": "d5b962d4bf30090d352de8445b32483491507a71a71b6708a86c68f18623130a",
        },
    },
    "got_a_talent": {
        "repo": "pollen-robotics/microduck-got-a-talent-reference",
        "revision": "0454fe8a848526338ba6edf72b8358fbf51d6b21",
        "files": {
            "policy.onnx": "332aeaccc5a9b9963e3aa097faa6ea7de325647d17da013f2482cd7e672a93cb",
            "manifest.json": "f6b188ba6a098c3cb1179cbd0be9d344a0725c9b5bfe7e8ebf47d3c71c6946d4",
            "timeline.json": "d4f7e3c0435942c0033371ebf5394379effb326f3fe4e89a6ec74d9a566ca01b",
        },
    },
    "sock_heist_6m": {
        "repo": "pollen-robotics/microduck-sock-heist-6m-reference",
        "revision": "ee81c1368572753528c01ee493c938c6aaac3ed0",
        "files": {
            "policy.onnx": "4325ed736009b307cf1eea81794b8088f4d8d9243dd77202405ef0ae379a5ec9",
            "manifest.json": "5b25a52443a9f6ef39132a855ed5b011206450c8beea35b17e39006c46706a6d",
        },
    },
}
