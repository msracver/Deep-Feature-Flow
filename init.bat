cd /d %~dp0
mkdir .\output
mkdir .\external\mxnet
mkdir .\model\pretrained_model
pause
cd lib\bbox
python setup_windows.py build_ext --inplace
cd ..\nms
python setup_windows.py build_ext --inplace
python setup_windows_cuda.py build_ext --inplace
cd ..\..
pause
