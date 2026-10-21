"""Wire the factorized planner (navsim/agents/para_ssr/fact/) into PARA-SSR.

Run from the repo root.  Every replacement asserts its anchor exists exactly
once, so a drifted file fails loudly instead of being half-patched.
"""
import pathlib
import sys

ROOT = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else ".").resolve()


def patch(rel, pairs):
    p = ROOT / rel
    s = p.read_text()
    for old, new in pairs:
        n = s.count(old)
        if n != 1:
            raise SystemExit(f"{rel}: anchor found {n} times:\n{old[:200]}")
        s = s.replace(old, new)
    p.write_text(s)
    print("patched", rel)


# ---------------------------------------------------------------- planner head
patch("navsim/agents/para_ssr/modules/planner_head.py", [
    ("""        plan_det_memory_mode: str = "mean",
        plan_map_memory_segments: int = 1,
    ):
        super().__init__()
        validate_plan_memory_options(plan_det_memory_mode, plan_map_memory_segments)""",
     """        plan_det_memory_mode: str = "mean",
        plan_map_memory_segments: int = 1,
        fact_planner: Optional[nn.Module] = None,
    ):
        super().__init__()
        validate_plan_memory_options(plan_det_memory_mode, plan_map_memory_segments)
        if fact_planner is not None and plan_anchor_file:
            raise ValueError("the factorized planner and the anchor planner are exclusive")"""),
    ("""        self.ego_fut_decoder = nn.Sequential(*reg_layers) if self.anchor_planner is None else None""",
     """        # v3: factorized path x speed vocabulary scored coarse-to-fine (fact/planner.py).
        # The planner layers then carry only the ego/scene token over the BEV and the
        # task memories; the candidates read the BEV themselves.
        self.fact_planner = fact_planner
        self.ego_fut_decoder = (nn.Sequential(*reg_layers)
                                if (self.anchor_planner is None and self.fact_planner is None) else None)"""),
    ("""        *,
        bev_pos: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        \"\"\"Plan from BEV and current [vx, vy, ax, ay] ego status.""",
     """        *,
        bev_pos: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
        fact_fields: Optional[Dict[str, torch.Tensor]] = None,
        fact_gt: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        \"\"\"Plan from BEV and current [vx, vy, ax, ay] ego status."""),
    ("""        h_ego = h
        if self.anchor_planner is not None:
            h = self.anchor_planner.queries(h)                      # [B, K, C]""",
     """        h_ego = h
        if self.fact_planner is not None:
            if fact_fields is None:
                raise ValueError("the factorized planner needs fact_fields (predicted sdf / occupancy)")
            for layer in self.planner_layers:
                h = layer(h, plan_pos, bev_embed, bev_pos, **memories)
            ego_tok = self.final_norm(h)[:, 0]                                        # [B, C]
            bev_map = bev_embed.transpose(1, 2).reshape(bs, self.embed_dims, self.bev_h, self.bev_w)
            gt_path = gt_speed = None
            if fact_gt is not None and self.training:
                gt_path, gt_speed = fact_gt.get("fact_gt_path"), fact_gt.get("fact_gt_speed")
            out = self.fact_planner(bev_map, ego_tok, ego_status, fact_fields["sdf"], fact_fields["occ"],
                                    use_extra=self.training, gt_path=gt_path, gt_speed=gt_speed)
            poses = out["trajectory"]
            steps = torch.diff(torch.cat([poses.new_zeros(bs, 1, self.traj_dims), poses], dim=1), dim=1)
            out.update(
                bev_embed=bev_embed, scene_query=h.transpose(0, 1), token_attn=None,
                ego_fut_preds=steps.unsqueeze(1).expand(bs, self.ego_fut_mode, self.fut_ts, self.traj_dims),
            )
            return out
        if self.anchor_planner is not None:
            h = self.anchor_planner.queries(h)                      # [B, K, C]"""),
])

# ---------------------------------------------------------------- model
patch("navsim/agents/para_ssr/para_ssr_model.py", [
    ("""from .modules.planner_head import ParaSSRPlannerHead""",
     """from .modules.planner_head import ParaSSRPlannerHead
from .fact.planner import FactorizedPlanner"""),
    ("""        self.pts_bbox_head = ParaSSRPlannerHead(
            transformer=transformer,""",
     """        fact_enable = bool(getattr(cfg, "fact_enable", False))
        fact_planner = None
        if fact_enable:
            if not cfg.fact_path_file or not cfg.fact_speed_file:
                raise ValueError("fact_enable=true needs fact_path_file and fact_speed_file")
            fact_planner = FactorizedPlanner(
                cfg.fact_path_file, cfg.fact_speed_file, cfg.pc_range,
                embed_dims=cfg.embed_dims, fut_ts=cfg.fut_ts, num_heads=cfg.num_heads,
                station_m=cfg.fact_station_m, num_stations=cfg.fact_num_stations,
                topk_path=cfg.fact_topk_path, topk_speed=cfg.fact_topk_speed, topk=cfg.fact_topk,
                im_weight=cfg.fact_im_weight, fine_layers=cfg.fact_fine_layers, dropout=cfg.fact_dropout,
                extra_anchor_file=(cfg.plan_anchor_file if cfg.fact_use_extra_anchors else None),
            )
            hidden = int(cfg.fact_aux_hidden)
            self.fact_sdf_head = nn.Sequential(
                nn.Conv2d(cfg.embed_dims, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, 1, 1))
            self.fact_occ_head = nn.Sequential(
                nn.Conv2d(cfg.embed_dims, hidden, 3, padding=1), nn.ReLU(inplace=True), nn.Conv2d(hidden, cfg.fut_ts, 1))
        else:
            self.fact_sdf_head = self.fact_occ_head = None
        self.pts_bbox_head = ParaSSRPlannerHead(
            transformer=transformer,
            fact_planner=fact_planner,"""),
    ("""            plan_anchor_file=getattr(cfg, "plan_anchor_file", None) if getattr(cfg, "plan_anchor", False) else None,""",
     """            plan_anchor_file=(getattr(cfg, "plan_anchor_file", None)
                              if (getattr(cfg, "plan_anchor", False) and not fact_enable) else None),"""),
    ("""    def forward(
        self, features: Dict[str, torch.Tensor], run_aux: Optional[bool] = None
    ) -> Dict[str, torch.Tensor]:""",
     """    def forward(
        self, features: Dict[str, torch.Tensor], run_aux: Optional[bool] = None,
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:"""),
    ("""        outs = self.pts_bbox_head.plan_from_bev(
            bev_embed, features["command"], det_out, map_out,
            # The cached status is command + (vx, vy, ax, ay) in NAVSIM's
            # native ego axes and metric units. Command has its own embedding.
            ego_status=features["status_feature"][:, cfg.num_navi_cmd:],
        )
        predictions: Dict[str, torch.Tensor] = {
            "bev_embed": bev_embed,
            "ego_fut_preds": outs["ego_fut_preds"],
            "trajectory": self.pts_bbox_head.select_trajectory(
                outs["ego_fut_preds"], features["command"]
            ),
        }""",
     """        fact_fields = fact_preds = None
        if self.fact_sdf_head is not None:
            B = bev_embed.shape[0]
            bev_map = bev_embed.transpose(1, 2).reshape(B, cfg.embed_dims, cfg.bev_h, cfg.bev_w)
            sdf_pred = torch.tanh(self.fact_sdf_head(bev_map)) * float(cfg.plan_map_clip)   # [B, 1, H, W] metres
            occ_logit = self.fact_occ_head(bev_map)                                          # [B, T, H, W]
            fact_fields = {"sdf": sdf_pred, "occ": torch.sigmoid(occ_logit)}
            fact_preds = {"fact_sdf_pred": sdf_pred, "fact_occ_logit": occ_logit, "fact_occ_pred": fact_fields["occ"]}
        outs = self.pts_bbox_head.plan_from_bev(
            bev_embed, features["command"], det_out, map_out,
            # The cached status is command + (vx, vy, ax, ay) in NAVSIM's
            # native ego axes and metric units. Command has its own embedding.
            ego_status=features["status_feature"][:, cfg.num_navi_cmd:],
            fact_fields=fact_fields, fact_gt=targets,
        )
        predictions: Dict[str, torch.Tensor] = {
            "bev_embed": bev_embed,
            "ego_fut_preds": outs["ego_fut_preds"],
            "trajectory": self.pts_bbox_head.select_trajectory(
                outs["ego_fut_preds"], features["command"]
            ),
        }
        if "fact_final" in outs:  # v3 factorized planner: its candidates, scores and the refined selection
            predictions["trajectory"] = outs["trajectory"]
            predictions.update({k: v for k, v in outs.items() if k.startswith(("fact_", "plan_")) or k == "sim_rewards"})
            predictions.update(fact_preds)"""),
])

# ---------------------------------------------------------------- loss
patch("navsim/agents/para_ssr/para_ssr_loss.py", [
    ("""from .modules.anchor_planner import anchor_plan_losses""",
     """from .modules.anchor_planner import anchor_plan_losses
from .fact.losses import fact_aux_losses, fact_plan_losses"""),
    ("""            logs.update({f"plan_v2/{k}": v.detach() for k, v in al.items()})
        # Keep the historical raw metric""",
     """            logs.update({f"plan_v2/{k}": v.detach() for k, v in al.items()})
        elif "fact_final" in predictions:
            # v3 factorized planner (fact/losses.py).  The L1 of the selected
            # trajectory stays in the logs only, as for v2.
            fl, fl_logs = fact_plan_losses(predictions, targets, features, cfg, model.pts_bbox_head.fact_planner)
            task_losses["plan"] = tw.get("plan", 1.0) * fl
            logs.update(fl_logs)
        # Keep the historical raw metric"""),
    ("""        # ---- readout-space distillation (report/19) ---------------------""",
     """        # ---- v3 auxiliary BEV heads: road SDF (map) and future occupancy (det) ----
        if "fact_sdf_pred" in predictions:
            aux = fact_aux_losses(predictions, targets, cfg)
            if "map" in aux:
                task_losses["map"] = task_losses.get("map", 0.0) + float(cfg.fact_sdf_weight) * aux["map"] * tw.get("map", 1.0)
                logs["loss_fact_sdf"] = aux["map"].detach()
            if "det" in aux:
                task_losses["det"] = task_losses.get("det", 0.0) + float(cfg.fact_occ_weight) * aux["det"] * tw.get("det", 1.0)
                logs["loss_fact_occ"] = aux["det"].detach()

        # ---- readout-space distillation (report/19) ---------------------"""),
])

# ---------------------------------------------------------------- agent
patch("navsim/agents/para_ssr/para_ssr_agent.py", [
    ("""from .plan_score_targets import AnchorScoreTargetBuilder""",
     """from .plan_score_targets import AnchorScoreTargetBuilder
from .fact.targets import FactTargetBuilder"""),
    ("""        if float(getattr(cfg, "plan_map_weight", 0.0)) > 0:
            builders.append(DrivableAreaTargetBuilder(cfg))
        # plan_score_file is training-only: evaluation leaves it null and gets no builder
        if getattr(cfg, "plan_anchor", False) and getattr(cfg, "plan_score_file", None):
            builders.append(AnchorScoreTargetBuilder(cfg))""",
     """        fact = bool(getattr(cfg, "fact_enable", False))
        if float(getattr(cfg, "plan_map_weight", 0.0)) > 0 or fact:
            builders.append(DrivableAreaTargetBuilder(cfg))
        if fact:
            builders.append(FactTargetBuilder(cfg))
        # plan_score_file is training-only: evaluation leaves it null and gets no builder
        wants_scores = getattr(cfg, "plan_anchor", False) or (fact and getattr(cfg, "fact_use_extra_anchors", False))
        if wants_scores and getattr(cfg, "plan_score_file", None):
            builders.append(AnchorScoreTargetBuilder(cfg))"""),
    ("""        self._loss.apply_aux_scales(self.para_ssr_model)
        predictions = self.para_ssr_model(features)""",
     """        self._loss.apply_aux_scales(self.para_ssr_model)
        predictions = self.para_ssr_model(features, targets=targets)"""),
    ("""        anchor_prefix = "para_ssr_model.pts_bbox_head.anchor_planner."
        if not any(k.startswith(anchor_prefix) for k in state_dict):
            fresh = [k for k in missing if k.startswith(anchor_prefix)]
            if fresh:
                logger.info("v1 checkpoint: %d anchor-planner tensors start fresh", len(fresh))
            missing = [k for k in missing if not k.startswith(anchor_prefix)]""",
     """        fresh_prefixes = (
            "para_ssr_model.pts_bbox_head.anchor_planner.",
            # v3 factorized planner and its two auxiliary BEV heads (warm start from a v1/v2 checkpoint)
            "para_ssr_model.pts_bbox_head.fact_planner.",
            "para_ssr_model.fact_sdf_head.",
            "para_ssr_model.fact_occ_head.",
        )
        for anchor_prefix in fresh_prefixes:
            if not any(k.startswith(anchor_prefix) for k in state_dict):
                fresh = [k for k in missing if k.startswith(anchor_prefix)]
                if fresh:
                    logger.info("checkpoint without %s: %d tensors start fresh", anchor_prefix, len(fresh))
                missing = [k for k in missing if not k.startswith(anchor_prefix)]"""),
    ("""        if getattr(config, "plan_anchor", False):
            if not config.plan_anchor_file:
                raise ValueError("plan_anchor=true needs plan_anchor_file")""",
     """        if getattr(config, "fact_enable", False):
            if not config.fact_path_file or not config.fact_speed_file:
                raise ValueError("fact_enable=true needs fact_path_file and fact_speed_file")
            if config.fact_use_extra_anchors and not config.plan_anchor_file:
                raise ValueError("fact_use_extra_anchors=true needs plan_anchor_file (the 256 WoTE anchors)")
            if not getattr(config, "input_target", False):
                raise ValueError("fact_enable=true needs input_target=true (the forced vocabulary indices)")
        if getattr(config, "plan_anchor", False):
            if not config.plan_anchor_file:
                raise ValueError("plan_anchor=true needs plan_anchor_file")"""),
])

# ---------------------------------------------------------------- config dataclass
patch("navsim/agents/para_ssr/configs/default.py", [
    ("""    plan_map_weight: float = 0.0
    plan_map_margin: float = 0.0""",
     """    # ---- v3 factorized planner (fact/): path x speed vocabulary, coarse-to-fine
    # PDM scoring with proxy labels + exact anchor labels, two auxiliary BEV heads.
    fact_enable: bool = False
    fact_path_file: Optional[str] = None       # [P, S+1, 2] path anchors (tools/fact/build_vocab.py)
    fact_speed_file: Optional[str] = None      # [V, T] cumulative-distance profiles
    fact_station_m: float = 1.0                # arc-length spacing of the path stations
    fact_num_stations: int = 16                # stations sampled per path in stage 1
    fact_topk_path: int = 20
    fact_topk_speed: int = 20
    fact_topk: int = 6                         # refined candidates returned
    fact_im_weight: float = 0.1                # selection: log PDMS + w * log softmax(im)
    fact_fine_layers: int = 2
    fact_dropout: float = 0.1
    fact_aux_hidden: int = 64
    fact_use_extra_anchors: bool = True        # score the 256 WoTE anchors with their exact labels (training)
    fact_route_stations: int = 128             # route centerline stations (1 m) for the EP proxy
    fact_route_spacing: float = 1.0
    # loss weights and temperatures (fact/losses.py)
    fact_w_path: float = 1.0
    fact_w_speed: float = 1.0
    fact_w_score: float = 1.0
    fact_w_list: float = 1.0
    fact_w_margin: float = 0.5
    fact_w_im: float = 1.0
    fact_w_offset: float = 1.0
    fact_w_extra: float = 1.0
    fact_tau_path: float = 1.0                 # m^2
    fact_tau_speed: float = 2.0                # m^2
    fact_tau_im: float = 1.0                   # m^2
    fact_tau_list: float = 0.05
    fact_margin: float = 2.0                   # logits
    fact_target_pdms: float = 0.95             # exact-label anchors above this are imitation targets
    fact_sdf_weight: float = 1.0               # aux road-SDF head (task map)
    fact_occ_weight: float = 1.0               # aux future-occupancy head (task det)
    fact_occ_pos_weight: float = 3.0

    plan_map_weight: float = 0.0
    plan_map_margin: float = 0.0"""),
])

# ---------------------------------------------------------------- agent yaml
patch("navsim/planning/script/config/common/agent/para_ssr_agent.yaml", [
    ("""  # Planning-side map consistency (plan_map.py); 0 = off.
  plan_map_weight: 0.0""",
     """  # v3 factorized planner (navsim/agents/para_ssr/fact/).  Off by default; see
  # scripts/training/train_fact.sh for the recipe (needs input_target: true).
  fact_enable: false
  fact_path_file: null
  fact_speed_file: null
  fact_station_m: 1.0
  fact_num_stations: 16
  fact_topk_path: 20
  fact_topk_speed: 20
  fact_topk: 6
  fact_im_weight: 0.1
  fact_fine_layers: 2
  fact_dropout: 0.1
  fact_aux_hidden: 64
  fact_use_extra_anchors: true
  fact_route_stations: 128
  fact_route_spacing: 1.0
  fact_w_path: 1.0
  fact_w_speed: 1.0
  fact_w_score: 1.0
  fact_w_list: 1.0
  fact_w_margin: 0.5
  fact_w_im: 1.0
  fact_w_offset: 1.0
  fact_w_extra: 1.0
  fact_tau_path: 1.0
  fact_tau_speed: 2.0
  fact_tau_im: 1.0
  fact_tau_list: 0.05
  fact_margin: 2.0
  fact_target_pdms: 0.95
  fact_sdf_weight: 1.0
  fact_occ_weight: 1.0
  fact_occ_pos_weight: 3.0
  # Planning-side map consistency (plan_map.py); 0 = off.
  plan_map_weight: 0.0"""),
])
print("done")
