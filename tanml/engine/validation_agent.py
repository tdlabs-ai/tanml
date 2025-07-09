from __future__ import annotations
from importlib.resources import files 
from datetime import datetime
import tzlocal
import json
from pathlib import Path

from tanml.engine.core_engine_agent import ValidationEngine
from tanml.report.report_builder import ReportBuilder
from tanml.engine.check_agent_registry import CHECK_RUNNER_REGISTRY


class SegmentValidator:
    def __init__(
        self,
        model,
        raw_df,
        cleaned_df,
        config,
        target_col=None,
        segment_column=None,
        segment_values=None,
        segment_name=None,
        report_base="reports"
    ):
        self.model = model
        self.raw_df = raw_df
        self.cleaned_df = cleaned_df
        self.config = config
        self.target_col = target_col
        self.segment_column = segment_column
        self.segment_values = segment_values
        self.segment_name = segment_name
        self.report_base = report_base

    def run(self):
        if self.segment_name:
            return self._run_single(self.cleaned_df, self.segment_name)

        if not self.segment_column:
            raise ValueError("Segmentation column not specified in rules.yaml")

        results = {}
        for segment in self.segment_values:
            segment_df = self.cleaned_df[self.cleaned_df[self.segment_column] == segment]
            print(f"🔹 Running validation for segment value: {segment}")
            result = self._run_single(segment_df, segment)
            results[segment] = result

        return results

    def _run_single(self, segment_df, segment_name):
        if self.target_col is None or self.target_col not in segment_df.columns:
            raise ValueError(f"❌ Target column '{self.target_col}' missing in cleaned data for segment '{segment_name}'")

        y = segment_df[self.target_col]
        cols_to_drop = [c for c in (self.target_col, self.segment_column) if c in segment_df.columns]
        X = segment_df.drop(columns=cols_to_drop)

        # Pass raw_df to ValidationEngine so CleaningReproCheck works
        engine = ValidationEngine(self.model, X, X, y, y, self.config, segment_df, self.raw_df)
        results = engine.run_all_checks()

        local_tz = tzlocal.get_localzone()
        now = datetime.now(local_tz)
        results["validation_date"] = now.strftime("%Y-%m-%d %H:%M:%S %Z (UTC%z)")
        results["model_path"] = self.model.__class__.__name__
        results["validated_by"] = "TanML Automated Validator"
        results["rules"] = self.config

        report_base_formatted = self.report_base.format(segment=segment_name) if "{segment}" in self.report_base else self.report_base
        output_dir = Path(report_base_formatted)
        output_dir.mkdir(parents=True, exist_ok=True)

        tpl_cfg = self.config.get("output", {}).get("template_path")     # may be None
        template_path = (
            Path(tpl_cfg).expanduser() if tpl_cfg
            else files("tanml.report.templates").joinpath("report_template.docx")
        )

        output_path = output_dir / f"report_{segment_name}.docx"

        builder = ReportBuilder(results, template_path, output_path)
        builder.build()

        print(f"📄 Report saved for segment '{segment_name}': {output_path}")
        print(json.dumps(results, indent=2, default=str))
        return results


__all__ = ["SegmentValidator", "CHECK_RUNNER_REGISTRY"]
