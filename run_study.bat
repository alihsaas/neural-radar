@echo off
cd /d "%~dp0"
uv run --locked --extra neural python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())" || goto fail
uv run --locked --extra neural pytest -q --basetemp=.pytest-tmp || goto fail
uv run --locked --extra neural python scripts/verify_baseline.py || goto fail
uv run --locked --extra neural python scripts/run_classical_benchmark.py || goto fail
uv run --locked --extra neural python scripts/train_detector.py --skip-existing || goto fail
uv run --locked --extra neural python scripts/evaluate_detector.py || goto fail
uv run --locked --extra neural python scripts/diagnose_neighbour_gain.py || goto fail
echo.
echo Finished. Results are in results\baseline, results\classical_benchmark and results\neural_detection.
pause
exit /b 0
:fail
echo.
echo A step failed. Read the message above.
pause
exit /b 1
