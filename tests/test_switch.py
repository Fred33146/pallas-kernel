from __future__ import annotations

import sys
import types

import pytest

from tops.switch import same_signature, is_env_true, get_env_val, use_native, switch_func
from tops.env import TOPS_NATIVE_ENV_NAME, TOPS_USE_NATIVE_MODULES, TOPS_USE_NATIVE_FUNCS


# ---------------------------------------------------------------------------
# same_signature
# ---------------------------------------------------------------------------

class TestSameSignature:
    def test_identical(self):
        def f(a, b, c=1): ...
        def g(a, b, c=1): ...
        assert same_signature(f, g) is True

    def test_different_names(self):
        def f(a, b): ...
        def g(x, y): ...
        assert same_signature(f, g) is False

    def test_different_defaults(self):
        def f(a, b=1): ...
        def g(a, b=2): ...
        assert same_signature(f, g) is False

    def test_different_count(self):
        def f(a): ...
        def g(a, b): ...
        assert same_signature(f, g) is False

    def test_no_params(self):
        def f(): ...
        def g(): ...
        assert same_signature(f, g) is True

    def test_kwargs(self):
        def f(a, **kw): ...
        def g(a, **kw): ...
        assert same_signature(f, g) is True


# ---------------------------------------------------------------------------
# is_env_true
# ---------------------------------------------------------------------------

class TestIsEnvTrue:
    @pytest.mark.parametrize("val", ["true", "TRUE", "True", "1"])
    def test_truthy(self, monkeypatch, val):
        monkeypatch.setenv("MY_FLAG", val)
        assert is_env_true("MY_FLAG") is True

    @pytest.mark.parametrize("val", ["false", "0", "no", ""])
    def test_falsy(self, monkeypatch, val):
        monkeypatch.setenv("MY_FLAG", val)
        assert is_env_true("MY_FLAG") is False

    def test_missing_env_defaults_false(self, monkeypatch):
        monkeypatch.delenv("MY_FLAG", raising=False)
        assert is_env_true("MY_FLAG") is False


# ---------------------------------------------------------------------------
# get_env_val
# ---------------------------------------------------------------------------

class TestGetEnvVal:
    def test_single(self, monkeypatch):
        monkeypatch.setenv("MY_LIST", "foo")
        assert get_env_val("MY_LIST") == ["foo"]

    def test_comma_separated(self, monkeypatch):
        monkeypatch.setenv("MY_LIST", "foo,bar,baz")
        assert get_env_val("MY_LIST") == ["foo", "bar", "baz"]

    def test_custom_sep(self, monkeypatch):
        monkeypatch.setenv("MY_LIST", "a;b;c")
        assert get_env_val("MY_LIST", sep=";") == ["a", "b", "c"]

    def test_uppercase_lowered(self, monkeypatch):
        monkeypatch.setenv("MY_LIST", "FOO,BAR")
        assert get_env_val("MY_LIST") == ["foo", "bar"]

    def test_missing_env_returns_empty(self, monkeypatch):
        monkeypatch.delenv("MY_LIST", raising=False)
        assert get_env_val("MY_LIST") == [""]


# ---------------------------------------------------------------------------
# use_native
# ---------------------------------------------------------------------------

class TestUseNative:
    def _make_func(self, name="my_func", module="test_switch"):
        def f(): ...
        f.__name__ = name
        f.__module__ = module
        return f

    def test_native_disabled(self, monkeypatch):
        """TOPS_NATIVE=false -> always False."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "0")
        assert use_native(self._make_func()) is False

    def test_native_global_no_filters(self, monkeypatch):
        """TOPS_NATIVE=true without filters -> globally enabled."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)
        assert use_native(self._make_func()) is True

    def test_func_in_native_funcs(self, monkeypatch):
        """TOPS_NATIVE=true + func name in TOPS_USE_NATIVE_FUNCS -> True."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "1")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.setenv(TOPS_USE_NATIVE_FUNCS, "my_func,other")
        assert use_native(self._make_func("my_func")) is True

    def test_func_not_in_native_funcs(self, monkeypatch):
        """TOPS_NATIVE=true + func name NOT in list -> False."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "1")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.setenv(TOPS_USE_NATIVE_FUNCS, "other_func")
        assert use_native(self._make_func("my_func")) is False

    def test_module_in_native_modules(self, monkeypatch):
        """TOPS_NATIVE=true + func.__module__ in TOPS_USE_NATIVE_MODULES -> True."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "1")
        monkeypatch.setenv(TOPS_USE_NATIVE_MODULES, "tops.ops.gla.chunk_h")
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)
        assert use_native(self._make_func("fwd", module="tops.ops.gla.chunk_h")) is True

    def test_module_not_in_native_modules(self, monkeypatch):
        """TOPS_NATIVE=true + func.__module__ NOT in TOPS_USE_NATIVE_MODULES -> False."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "1")
        monkeypatch.setenv(TOPS_USE_NATIVE_MODULES, "tops.ops.gla.chunk_h")
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)
        assert use_native(self._make_func("fwd", module="tops.ops.simple_gla")) is False


# ---------------------------------------------------------------------------
# Helpers for switch_func tests
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_modules():
    """Register fake modules in sys.modules and clean up after test."""
    registered = []

    def _register(module_path: str, attrs: dict):
        mod = types.ModuleType(module_path)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[module_path] = mod
        registered.append(module_path)
        return mod

    yield _register

    for name in registered:
        sys.modules.pop(name, None)


def _make_orig(module_path: str, name: str, sig_params="q, k, v"):
    """Create a function with given __module__ and __name__."""
    ns = {}
    exec(f"def {name}({sig_params}): return 'orig'", ns)
    fn = ns[name]
    fn.__module__ = module_path
    return fn


# ---------------------------------------------------------------------------
# switch_func — test with @switch_func decorator
# ---------------------------------------------------------------------------

class TestSwitchFuncDecorator:

    def test_switches_to_alt_when_native(self, monkeypatch, fake_modules):
        """@switch_func: TOPS_NATIVE=true -> switch to cpu alt implementation."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        def compute(q, k, v):
            return "alt_result"

        fake_modules("tops.cpu.fake.mod_a", {"compute": compute})

        @switch_func
        def compute(q, k, v):
            return "orig_result"

        compute.__module__ = "tops.fake.mod_a"

        # @switch_func executes at definition time, need to re-apply decorator
        switched = switch_func(compute)
        assert switched("q", "k", "v") == "alt_result"

    def test_keeps_orig_when_not_native(self, monkeypatch, fake_modules):
        """@switch_func: TOPS_NATIVE=false -> keep original implementation."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "false")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        def run(x):
            return "alt"

        fake_modules("tops.cpu.fake.mod_b", {"run": run})

        orig = _make_orig("tops.fake.mod_b", "run", "x")
        switched = switch_func(orig)
        assert switched("x") == "orig"

    def test_alt_func_not_found(self, monkeypatch, fake_modules):
        """No matching func in alt module -> raise AttributeError."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        fake_modules("tops.cpu.fake.mod_c", {})  # empty module

        orig = _make_orig("tops.fake.mod_c", "missing_fn")
        with pytest.raises(AttributeError):
            switch_func(orig)

    def test_signature_mismatch_raises(self, monkeypatch, fake_modules):
        """Mismatched alt signature -> raise TypeError."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        def fwd(a, b, c, d):  # 4 params vs 2
            return "alt"

        fake_modules("tops.cpu.fake.mod_d", {"fwd": fwd})

        orig = _make_orig("tops.fake.mod_d", "fwd", "a, b")
        with pytest.raises(TypeError):
            switch_func(orig)

    def test_module_path_transform(self, monkeypatch, fake_modules):
        """tops.ops.simple_gla.chunk_h -> tops.cpu.ops.simple_gla.chunk_h."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        def chunk_fwd_h(q, k, v):
            return "cpu_impl"

        fake_modules("tops.cpu.ops.simple_gla.chunk_h", {"chunk_fwd_h": chunk_fwd_h})

        orig = _make_orig("tops.ops.simple_gla.chunk_h", "chunk_fwd_h")
        switched = switch_func(orig)
        assert switched("q", "k", "v") == "cpu_impl"

    def test_switch_by_func_name_filter(self, monkeypatch, fake_modules):
        """TOPS_NATIVE=true + TOPS_USE_NATIVE_FUNCS filter switches by func name."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.setenv(TOPS_USE_NATIVE_FUNCS, "target_fn")

        def target_fn(x):
            return "alt"

        fake_modules("tops.cpu.fake.mod_e", {"target_fn": target_fn})

        orig = _make_orig("tops.fake.mod_e", "target_fn", "x")
        switched = switch_func(orig)
        assert switched("x") == "alt"

    def test_func_not_in_filter_keeps_orig(self, monkeypatch, fake_modules):
        """TOPS_NATIVE=true but func not in filter list -> keep original."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.setenv(TOPS_USE_NATIVE_FUNCS, "other_fn")

        def my_fn(x):
            return "alt"

        fake_modules("tops.cpu.fake.mod_f", {"my_fn": my_fn})

        orig = _make_orig("tops.fake.mod_f", "my_fn", "x")
        switched = switch_func(orig)
        assert switched("x") == "orig"

    def test_non_tops_module_keeps_orig(self, monkeypatch):
        """Module not starting with 'tops.' -> return original immediately."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")
        monkeypatch.delenv(TOPS_USE_NATIVE_MODULES, raising=False)
        monkeypatch.delenv(TOPS_USE_NATIVE_FUNCS, raising=False)

        orig = _make_orig("other_pkg.ops.module", "my_fn", "x")
        switched = switch_func(orig)
        assert switched is orig

    def test_alt_module_import_fails(self, monkeypatch):
        """Alt module does not exist -> raise ModuleNotFoundError."""
        monkeypatch.setenv(TOPS_NATIVE_ENV_NAME, "true")

        orig = _make_orig("tops.nonexistent.module", "func_a")
        with pytest.raises(ModuleNotFoundError):
            switch_func(orig)
