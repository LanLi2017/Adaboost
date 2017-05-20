#u Atemp

步骤：
    初始化
        cmd进入该文件夹，进入Atemp文件夹里面，然后初始化：git init（将生成一个隐藏文件.git）
        然后git remote add origin https://github.com/LanLi2017/Atemp.git 实现本地和github上的联系
        然后将本地和github上的版本合并：git pull origin master（下载到本地，并合并）

    写代码阶段
        做好自己任务后，先commit自己的代码，然后从git服务器下载合并代码到本地，然后将自己的代码上传合并到git
        在Atemp文件夹下，依次执行：
            1. git add -A       (配合commit使用，先这个再commit)
            2. git commit -m "description something"  （在” “ 中可以写一些注释，即commit的代码有哪些，改动有哪些）
            3. git pull origin master   （下载合并版本到本地（必须先commit）。
            4. git push origin master   （上传代码到git服务器上。





