### 批量处理STM数据

目前只完成了stm文件概览图生成，后续可能结合NOMAD或者AI做数据处理。

使用概览图生成功能下载 release v1.0.


STM文件比较多时查阅目标数据文件不易，给每个文件夹生成概览图。
主要是AI写的，我调教了gwyfile和nanonispy2这两个包的用法，AI似乎尚不熟悉。仅此留存。

第一步：创建环境。在文件所在目录打开终端，输入执行：

conda env create -f environment.yml

这个命令会根据 environment.yml文件自动创建新环境并安装所有依赖。

第三步：打开文件“Auto_overview_generate.ipynb”，根据提示运行代码。
