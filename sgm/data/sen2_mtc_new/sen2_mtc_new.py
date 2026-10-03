import os
import glob
import numpy as np
import tifffile as tiff
import torch
from torch.utils.data import Dataset


class Sen2_MTC_New_Multi(Dataset):
    def __init__(
        self,
        data_root,
        use_ir=True,
        mode="train",
        mono_temporal=-1,
        multi_temporal=3,
        selector_soft_root=None,
        selector_soft_required=True,
        selector_conf_required=True,
        selector_soft_mode="none",
    ):
        """
        selector_soft_mode 支持四种模式:
        - "none": baseline，不使用 soft / conf
        - "temporal": 当前旧方案，soft 作为第 4 个伪时相，同时进入 cond_image 和 raw_image(mu)
        - "condition_only_soft": A1，soft 只作为 condition，形式为 [cloud_i, soft]
        - "condition_only_soft_conf": A2，soft+conf 只作为 condition，形式为 [cloud_i, soft, conf]
        - "condition_only_masked_residual_hint": A3，masked residual hint 只作为 condition，形式为 [cloud_i, m*(soft-cloud_i)]

        注意:
        - temporal 模式要求 multi_temporal=4
        - condition_only_soft / condition_only_soft_conf / condition_only_masked_residual_hint 要求 multi_temporal=3
        - raw_image 始终只保留 RGB 三通道
        - use_ir=True 时，cloud/soft 的 condition 通道为 4 通道；否则为 3 通道
        """
        self.data_root = data_root
        self.mode = mode
        self.use_ir = use_ir
        self.filepair = []
        self.image_name = []
        self.mono_temporal = mono_temporal

        assert multi_temporal in [2, 3, 4], "multi_temporal should be 2, 3, or 4"
        self.multi_temporal = multi_temporal

        assert selector_soft_mode in [
            "none",
            "temporal",
            "condition_only_soft",
            "condition_only_soft_conf",
            "condition_only_masked_residual_hint",
        ], (
            "selector_soft_mode must be one of: "
            "none, temporal, condition_only_soft, condition_only_soft_conf, condition_only_masked_residual_hint"
        )
        self.selector_soft_mode = selector_soft_mode
        self.selector_soft_root = selector_soft_root
        self.selector_soft_required = selector_soft_required
        self.selector_conf_required = selector_conf_required

        self.use_selector_soft = selector_soft_mode in [
            "temporal",
            "condition_only_soft",
            "condition_only_soft_conf",
            "condition_only_masked_residual_hint",
        ]
        self.use_selector_conf = selector_soft_mode == "condition_only_soft_conf"

        # 兼容: 旧代码若只把 multi_temporal 设为 4，则默认视为 temporal 模式
        if self.multi_temporal == 4 and self.selector_soft_mode == "none":
            self.selector_soft_mode = "temporal"
            self.use_selector_soft = True

        if self.selector_soft_mode == "temporal" and self.multi_temporal != 4:
            raise ValueError("selector_soft_mode='temporal' 时，multi_temporal 必须为 4")

        if self.selector_soft_mode in ["condition_only_soft", "condition_only_soft_conf", "condition_only_masked_residual_hint"] and self.multi_temporal != 3:
            raise ValueError(
                "condition-only 方案时，multi_temporal 必须为 3；"
                "因为 soft 不再作为第 4 个 raw_image / mu 时相"
            )

        if self.use_selector_soft and self.selector_soft_root is None:
            raise ValueError("使用 selector_soft 时必须提供 selector_soft_root")

        if mode == "train":
            list_name = "train.txt"
        elif mode == "val":
            list_name = "val.txt"
        elif mode == "test":
            list_name = "test.txt"
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        list_path = os.path.join(self.data_root, list_name)
        if not os.path.exists(list_path):
            raise RuntimeError(f"split txt not found: {list_path}")

        self.tile_list = np.loadtxt(list_path, dtype=str)
        self.tile_list = np.atleast_1d(self.tile_list).tolist()

        for tile in self.tile_list:
            cloudless_dir = os.path.join(self.data_root, "Sen2_MTC", tile, "cloudless")
            cloud_dir = os.path.join(self.data_root, "Sen2_MTC", tile, "cloud")

            if not os.path.isdir(cloudless_dir):
                raise RuntimeError(f"cloudless dir not found: {cloudless_dir}")
            if not os.path.isdir(cloud_dir):
                raise RuntimeError(f"cloud dir not found: {cloud_dir}")

            image_name_list = sorted(
                [
                    os.path.splitext(name)[0]
                    for name in os.listdir(cloudless_dir)
                    if name.endswith(".tif")
                ]
            )

            for image_name in image_name_list:
                image_cloud_path0 = os.path.join(cloud_dir, image_name + "_0.tif")
                image_cloud_path1 = os.path.join(cloud_dir, image_name + "_1.tif")
                image_cloud_path2 = os.path.join(cloud_dir, image_name + "_2.tif")
                image_cloudless_path = os.path.join(cloudless_dir, image_name + ".tif")

                paths = [
                    image_cloud_path0,
                    image_cloud_path1,
                    image_cloud_path2,
                    image_cloudless_path,
                ]
                for p in paths:
                    if not os.path.exists(p):
                        raise RuntimeError(f"image file not found: {p}")

                image_soft_path = None
                image_conf_path = None

                if self.use_selector_soft:
                    image_soft_path = self.find_selector_soft_path(tile, image_name)
                    if image_soft_path is None:
                        msg = (
                            f"selector_soft not found for tile={tile}, image_name={image_name}, "
                            f"selector_soft_root={self.selector_soft_root}"
                        )
                        if self.selector_soft_required:
                            raise RuntimeError(msg)
                        else:
                            print("[WARN]", msg)
                            continue

                if self.use_selector_conf:
                    image_conf_path = self.find_selector_conf_path(tile, image_name)
                    if image_conf_path is None:
                        msg = (
                            f"selector_conf not found for tile={tile}, image_name={image_name}, "
                            f"selector_soft_root={self.selector_soft_root}"
                        )
                        if self.selector_conf_required:
                            raise RuntimeError(msg)
                        else:
                            print("[WARN]", msg)
                            continue

                self.filepair.append(
                    {
                        "cloud0": image_cloud_path0,
                        "cloud1": image_cloud_path1,
                        "cloud2": image_cloud_path2,
                        "soft": image_soft_path,
                        "conf": image_conf_path,
                        "gt": image_cloudless_path,
                        "tile": tile,
                        "image_name": image_name,
                    }
                )
                self.image_name.append(image_name)

        self.augment_rotation_param = np.random.randint(0, 4, len(self.filepair))
        self.augment_flip_param = np.random.randint(0, 3, len(self.filepair))

        print(
            f"[Sen2_MTC_New_Multi] mode={self.mode}, "
            f"num_samples={len(self.filepair)}, "
            f"multi_temporal={self.multi_temporal}, "
            f"selector_soft_mode={self.selector_soft_mode}, "
            f"selector_soft_root={self.selector_soft_root}"
        )

    def __len__(self):
        return len(self.filepair)

    def find_selector_soft_path(self, tile, image_name):
        if self.selector_soft_root is None:
            return None

        root = self.selector_soft_root
        candidates = [
            os.path.join(root, tile, "selector_soft", image_name + ".tif"),
            os.path.join(root, tile, "selector_soft", image_name + "_selector_soft.tif"),
            os.path.join(root, tile, image_name + "_selector_soft.tif"),
            os.path.join(root, tile, image_name + ".tif"),
            os.path.join(root, tile, image_name, "selector_soft.tif"),
            os.path.join(root, tile + "__" + image_name, "selector_soft.tif"),
            os.path.join(root, tile + "__" + image_name + "_selector_soft.tif"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        patterns = [
            os.path.join(root, "**", tile, "selector_soft", image_name + ".tif"),
            os.path.join(root, "**", "selector_soft", image_name + ".tif"),
            os.path.join(root, "**", image_name + "*soft*.tif"),
        ]
        for pattern in patterns:
            matched = sorted(glob.glob(pattern, recursive=True))
            if len(matched) > 0:
                return matched[0]
        return None

    def find_selector_conf_path(self, tile, image_name):
        if self.selector_soft_root is None:
            return None

        root = self.selector_soft_root
        candidates = [
            os.path.join(root, tile, "selector_conf", image_name + "_conf.npy"),
            os.path.join(root, tile, "selector_conf", image_name + ".npy"),
            os.path.join(root, tile, image_name + "_conf.npy"),
            os.path.join(root, tile + "__" + image_name, "selector_conf.npy"),
        ]
        for p in candidates:
            if os.path.exists(p):
                return p

        patterns = [
            os.path.join(root, "**", tile, "selector_conf", image_name + "_conf.npy"),
            os.path.join(root, "**", "selector_conf", image_name + "_conf.npy"),
            os.path.join(root, "**", image_name + "*conf*.npy"),
        ]
        for pattern in patterns:
            matched = sorted(glob.glob(pattern, recursive=True))
            if len(matched) > 0:
                return matched[0]
        return None


    def build_masked_residual_hint(self, cloud_cond, soft_cond, eps=1e-6):
        """
        Build masked residual hint in normalized [-1, 1] space.

        cloud_cond: [C,H,W], cloudy condition image.
        soft_cond:  [C,H,W], selector soft image, same channels as cloud_cond.

        diff = soft - cloud keeps the correction direction.
        m = normalized mean(abs(diff)) is a single-channel spatial gate indicating
        where soft and cloudy disagree strongly.
        hint = m * diff suppresses unreliable guidance in low-difference/clear areas.
        """
        diff = soft_cond - cloud_cond
        m = torch.mean(torch.abs(diff), dim=0, keepdim=True)
        m_min = torch.amin(m, dim=(1, 2), keepdim=True)
        m_max = torch.amax(m, dim=(1, 2), keepdim=True)
        m = (m - m_min) / (m_max - m_min + eps)
        hint = diff * m
        return hint

    def __getitem__(self, index):
        item = self.filepair[index]

        cloud_image_path0 = item["cloud0"]
        cloud_image_path1 = item["cloud1"]
        cloud_image_path2 = item["cloud2"]
        cloudless_image_path = item["gt"]
        selector_soft_path = item["soft"]
        selector_conf_path = item["conf"]

        image_cloud0 = self.image_read(cloud_image_path0, index)
        image_cloud1 = self.image_read(cloud_image_path1, index)
        image_cloud2 = self.image_read(cloud_image_path2, index)
        image_cloudless = self.image_read(cloudless_image_path, index)

        image_selector_soft = None
        image_selector_conf = None
        if self.use_selector_soft:
            image_selector_soft = self.image_read(selector_soft_path, index)
        if self.use_selector_conf:
            image_selector_conf = self.read_confidence(selector_conf_path, index)

        # raw_image 始终只放 RGB
        raw_list = [
            image_cloud0[:3, :, :],
            image_cloud1[:3, :, :],
            image_cloud2[:3, :, :],
        ]

        # cloud 本身的 condition
        if self.use_ir:
            cloud_cond_list = [image_cloud0, image_cloud1, image_cloud2]  # each [4,H,W]
        else:
            cloud_cond_list = [
                image_cloud0[:3, :, :],
                image_cloud1[:3, :, :],
                image_cloud2[:3, :, :],
            ]

        if self.selector_soft_mode == "none":
            cond_list = cloud_cond_list

        elif self.selector_soft_mode == "temporal":
            cond_list = list(cloud_cond_list)
            if self.use_ir:
                cond_list.append(image_selector_soft)              # [4,H,W]
            else:
                cond_list.append(image_selector_soft[:3, :, :])   # [3,H,W]
            raw_list.append(image_selector_soft[:3, :, :])        # 第 4 个 mu 时相

        elif self.selector_soft_mode == "condition_only_soft":
            if self.use_ir:
                soft_cond = image_selector_soft                    # [4,H,W]
            else:
                soft_cond = image_selector_soft[:3, :, :]         # [3,H,W]
            cond_list = [
                torch.cat([cloud_cond_list[0], soft_cond], dim=0),
                torch.cat([cloud_cond_list[1], soft_cond], dim=0),
                torch.cat([cloud_cond_list[2], soft_cond], dim=0),
            ]

        elif self.selector_soft_mode == "condition_only_soft_conf":
            if self.use_ir:
                soft_cond = image_selector_soft                    # [4,H,W]
            else:
                soft_cond = image_selector_soft[:3, :, :]         # [3,H,W]
            conf_cond = image_selector_conf                        # [1,H,W], already normalized to [-1,1]
            cond_list = [
                torch.cat([cloud_cond_list[0], soft_cond, conf_cond], dim=0),
                torch.cat([cloud_cond_list[1], soft_cond, conf_cond], dim=0),
                torch.cat([cloud_cond_list[2], soft_cond, conf_cond], dim=0),
            ]

        elif self.selector_soft_mode == "condition_only_masked_residual_hint":
            if self.use_ir:
                soft_cond = image_selector_soft                    # [4,H,W], normalized to [-1,1]
            else:
                soft_cond = image_selector_soft[:3, :, :]         # [3,H,W], normalized to [-1,1]

            hint0 = self.build_masked_residual_hint(cloud_cond_list[0], soft_cond)
            hint1 = self.build_masked_residual_hint(cloud_cond_list[1], soft_cond)
            hint2 = self.build_masked_residual_hint(cloud_cond_list[2], soft_cond)

            cond_list = [
                torch.cat([cloud_cond_list[0], hint0], dim=0),
                torch.cat([cloud_cond_list[1], hint1], dim=0),
                torch.cat([cloud_cond_list[2], hint2], dim=0),
            ]

        else:
            raise ValueError(f"Unknown selector_soft_mode: {self.selector_soft_mode}")

        ret = {}
        ret["gt_image"] = image_cloudless[:3, :, :]
        ret["cond_image"] = torch.stack(cond_list, dim=0)
        ret["raw_image"] = torch.stack(raw_list, dim=0)
        ret["path"] = item["image_name"] + ".png"
        ret["case_id"] = item["tile"] + "__" + item["image_name"]
        ret["dataset_name"] = "sen2_mtc_new"

        if self.mono_temporal != -1:
            ret["cond_image"] = ret["cond_image"][self.mono_temporal]
            ret["raw_image"] = ret["raw_image"][self.mono_temporal]
        else:
            ret["cond_image"] = ret["cond_image"][:self.multi_temporal]
            ret["raw_image"] = ret["raw_image"][:self.multi_temporal]

        return ret

    def image_read(self, image_path, sample_index=None):
        img = tiff.imread(image_path)

        if img.ndim == 2:
            img = img[None, :, :]
        elif img.ndim == 3:
            if img.shape[0] in [1, 3, 4] and img.shape[-1] not in [1, 3, 4]:
                pass
            else:
                img = img.transpose((2, 0, 1))
        else:
            raise ValueError(f"Unsupported image shape: {img.shape}, path={image_path}")

        img = img.astype(np.float32)

        if self.mode == "train" and sample_index is not None:
            flip_param = self.augment_flip_param[sample_index]
            rotation_param = self.augment_rotation_param[sample_index]

            if flip_param != 0:
                img = np.flip(img, flip_param)
            if rotation_param != 0:
                img = np.rot90(img, rotation_param, (1, 2))

        image = torch.from_numpy(img.copy()).float()
        image = image / 10000.0

        mean = torch.full((image.shape[0],), 0.5, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        std = torch.full((image.shape[0],), 0.5, dtype=image.dtype, device=image.device).view(-1, 1, 1)
        image.sub_(mean).div_(std)
        return image

    def read_confidence(self, conf_path, sample_index=None):
        conf = np.load(conf_path).astype(np.float32)

        if conf.ndim == 2:
            conf = conf[None, :, :]
        elif conf.ndim == 3:
            if conf.shape[0] == 1:
                pass
            elif conf.shape[-1] == 1:
                conf = conf.transpose((2, 0, 1))
            else:
                raise ValueError(f"Unsupported conf shape: {conf.shape}, path={conf_path}")
        else:
            raise ValueError(f"Unsupported conf shape: {conf.shape}, path={conf_path}")

        if self.mode == "train" and sample_index is not None:
            flip_param = self.augment_flip_param[sample_index]
            rotation_param = self.augment_rotation_param[sample_index]

            if flip_param != 0:
                conf = np.flip(conf, flip_param)
            if rotation_param != 0:
                conf = np.rot90(conf, rotation_param, (1, 2))

        conf = torch.from_numpy(conf.copy()).float()
        conf = torch.clamp(conf, 0.0, 1.0)
        conf = conf * 2.0 - 1.0
        return conf
