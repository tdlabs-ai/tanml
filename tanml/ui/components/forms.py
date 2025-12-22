# tanml/ui/components/forms.py
"""
Configuration forms and input widgets for TanML UI.
"""

from __future__ import annotations

import streamlit as st
from tanml.models.registry import (
    list_models, ui_schema_for, get_spec, infer_task_from_target
)

def render_model_form(y_train, seed_global: int, target_name: str = "default"):
    """Return (library, algorithm, params, task) using the 20-model registry,
    but never show per-model seed; we inject sidebar seed automatically.
    """
    task_auto = infer_task_from_target(y_train)
    task = st.radio(
        "Task",
        ["classification", "regression"],
        index=0 if task_auto == "classification" else 1,
        horizontal=True,
        key=f"mdl_task_{target_name}_salted"
    )

    libraries_all = ["sklearn", "xgboost", "lightgbm", "catboost"]
    library = st.selectbox("Library", libraries_all, index=0, key="mdl_lib")

    avail = [(lib, algo) for (lib, algo), spec in list_models(task).items() if lib == library]
    if not avail:
        st.error(f"No algorithms available for {library} / {task}. Is the library installed?")
        st.stop()
    algo_names = [a for (_, a) in avail]
    algo = st.selectbox("Algorithm", algo_names, index=0, key="mdl_algo")

    spec = get_spec(library, algo)
    schema = ui_schema_for(library, algo)
    defaults = spec.defaults or {}

    seed_keys = [k for k in ("random_state", "seed", "random_seed") if k in defaults]
    params = {}

    with st.expander("Hyperparameters", expanded=True):
        # Seed input - let user choose
        if seed_keys:
            seed_key = seed_keys[0]  # Use the first matching seed key
            default_seed = st.session_state.get("global_seed", 42)
            user_seed = st.number_input(
                f"🎲 Random Seed ({seed_key})", 
                min_value=0, 
                max_value=999999, 
                value=default_seed, 
                step=1,
                help="Set the random seed for reproducibility. Change this to get different model results."
            )
            params[seed_key] = int(user_seed)

        # Custom 2-Column Grid for Params
        c1, c2 = st.columns(2)
        
        # Filter out seed keys first to keep indexing clean
        valid_items = {k: v for k, v in schema.items() if k not in seed_keys}
        
        for i, (name, (typ, choices, helptext)) in enumerate(valid_items.items()):
            col = c1 if i % 2 == 0 else c2
            
            with col:
                default_val = defaults.get(name)

                if typ == "choice":
                    opts = list(choices) if choices else []
                    show = ["None" if o is None else o for o in opts]
                    if show:
                        if default_val is None and "None" in show:
                            idx = show.index("None")
                        elif default_val in show:
                            idx = show.index(default_val)
                        else:
                            idx = 0
                        sel = st.selectbox(name, show, index=idx, help=helptext)
                        params[name] = None if sel == "None" else sel
                    else:
                        params[name] = st.text_input(
                            name,
                            value=str(default_val) if default_val is not None else "",
                            help=helptext
                        )

                elif typ == "bool":
                    params[name] = st.checkbox(
                        name,
                        value=bool(default_val) if default_val is not None else False,
                        help=helptext
                    )

                elif typ == "int":
                    params[name] = int(st.number_input(
                        name,
                        value=int(default_val) if default_val is not None else 0,
                        step=1,
                        help=helptext
                    ))

                elif typ == "float":
                    params[name] = float(st.number_input(
                        name,
                        value=float(default_val) if default_val is not None else 0.0,
                        help=helptext
                    ))

                else:  # "str"
                    params[name] = st.text_input(
                        name,
                        value=str(default_val) if default_val is not None else "",
                        help=helptext
                    )

    for k in seed_keys:
        params[k] = int(seed_global)
        
    return library, algo, params, task
