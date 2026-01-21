# 先上效果图

![登录](https://www.zcec.top/usr/uploads/2026/01/2448730740.png)
![选课](https://www.zcec.top/usr/uploads/2026/01/1251060327.png)

# 使用方法

1. 准备：下载仓库，可用git clone或者去网页上下载zip
2. 前提条件：你有一个支持openai调用且支持 **图片识别** 的大模型（可能需付费），例如[阿里云百炼](https://bailian.console.aliyun.com)
3. 安装Python **3.12** （或以下），并安装依赖：
   可以使用
   
   ```
   pip install requests urllib3 pycryptodome serverchan-sdk pillow numpy ultralytics openai
   # 如果要用代理还需要下面这个
   pip install "requests[socks]"
   ```
   
   或者用requirements.txt（在你下载这个项目的目录里执行）
   
   ```
   pip install -r requirements.txt
   ```
4. 在校内网络或使用[第三方EasyConnect实现](https://github.com/lyc8503/NJUConnect)
5. 修改xk.conf
   请按照说明修改模板，不需要的地方**把中文删掉留空**

| 选项 | 作用 | 是否必填 |
| :---: | :---: | :---: |
| USER | 学号 | √ |
| PWD | 明文密码(已弃用) | × |
| PWD_ENCRYPT | 加密后密码 | √ |
| LLM_KEY | 大模型API密钥 | √ |
| LLM_MODEL | 模型(如qwen-vl-max) | √ |
| LLM_BASE_URL | 大模型API地址 | √ |
| MAX_RETRIES | 最大登录重试次数(建议为10) | √ |
| SCT_KEY | ServerChan的Key(推送用) | × |
| SCT_OPTIONS | 建议保持不动 | × |
| PROXY | 代理服务器 | × |
| CJY_USER | 不提供 | × |
| CJY_PASS | 最后三个可以直接删掉 | × |
| CJY_SOFTID | 一般用不到 | × |

加密后密码获取方式：
在选课平台按F12，选Network，填好信息登录，看图
![获取密码](https://www.zcec.top/usr/uploads/2026/01/3740882447.png)
把loginPwd后面那一串贴进去就行

6. 获取选课批次
   
   ```
   python get_batch_code.py
   ```
7. 设置课程：
   运行解密脚本

```
python decrypt.py
```

建议直接去抓选课请求 ~~(或者根据文末的提示去猜)~~ ：
点击选课后，会看到volunteer.do的请求，点开Payload，复制那一坨东西
![请求](https://www.zcec.top/usr/uploads/2026/01/3464656477.png)
扔到解密脚本里，注意ClassID、Kind、Type
![解密](https://www.zcec.top/usr/uploads/2026/01/949162351.png)

8. 修改course.conf

```
{
"electiveBatchCode": "正常情况下这里已经填好了，不要动",
  "courses": [
    ["ClassID1", "KIND1", "TYPE1"],
    ["ClassID2", "KIND2", "TYPE2"]
  ]
}
```

按照这样填上你要的课程，注意除最后一个课程外，行末有英文逗号

## 附：关于ClassID和Kind的一些猜测

~~可以根据这个结合初选时看到的东西，在退补选开始时快速抢课~~

ClassID应该是学期（例如202520262表示2025-2026学年第二学期）+课程号+第几个班级
Kind有这样一个对照表：

| 中文名         | TYPE | KIND       | 初选是否存在 |
| :----------------: | :------: | :------------: | :------: |
| 公共           | GG   | `-`    |-|
| 公选课         | GG01 | `4`    |√|
| 导学/研讨/通识 | GG02 | `6,7`  |√|
| 交流生语言课   | GG03 | `9`    |?|
| 国际化课程     | GG04 | `10`   |?|
| 其他国际化课程 | GG05 | `20`   |?|
| 科学之光       | GG06 | `3`    |√|
| 美育           | MY   | `5`    |√|
| 体育           | TY   | `2`    |√|
| 跨专业         | KZY  | `12`   |√|
| 悦读           | YD   | `8`    |√|
| 通修           | TX   | `-`    |-|
| 大学数学       | TX01 | `13`   |×|
| 大学英语       | TX02 | `14`   |×|
| 思政军事类     | TX03 | `15`   |×|
| 计算机         | TX04 | `16`   |×|
| 收藏           | SC   | `null` |×|
| 课表查询       | QB   | `null` |√|
| 专业           | ZY   | `1`    |√|
