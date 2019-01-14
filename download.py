# coding=utf-8
"""根据搜索词下载百度图片"""
import re
import urllib
import requests
import os

filePath='/Users/wangyuhui/Desktop/'
keyword = '水仙花'  # 关键词, 改为你想输入的词即可, 相当于在百度图片里搜索一样
url_init_first = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word='
url_init = url_init_first + urllib.quote(keyword, safe='/') + '&pn=0'

def prepare_file_system():
  if os.path.exists(os.path.join(filePath,keyword)):
      os.removedirs(os.path.join(filePath,keyword))
  os.makedirs(os.path.join(filePath,keyword))

def get_onePage_urls(onePageurl):
    """获取单个翻页的所有图片的urls+当前翻页的下一翻页的url"""
    if not onePageurl:
        print('已到最后一页, 结束')
        return [], ''
    try:
        html = requests.get(onePageurl).text
    except Exception as e:
        print(e)
        pic_urls = []
        next_url = ''
        return pic_urls, next_url
    pic_urls = re.findall('"objURL":"(.*?)",', html, re.S)#re.S表示.也可以表示\n
    for pul in pic_urls:
        print "****"+pul
    next_urls = re.findall(re.compile(r'<a href="(.*)" class="n">下一页</a>'), html, flags=0)
    next_url = 'https://image.baidu.com'+next_urls[0]
    print "得到的下一页地址为",next_url
    return pic_urls, next_url


def down_pic(pic_urls):
    """给出图片链接列表, 下载所有图片"""
    for i, pic_url in enumerate(pic_urls):
        try:
            pic = requests.get(pic_url, timeout=30)
            string = filePath + keyword + '/' + str(i + 1) + '.jpg'
            """with open(string, 'wb') as f:
                f.write(pic.content)
                print('成功下载第%s张图片: %s' % (str(i + 1), str(pic_url)))"""
        except Exception as e:
            print('下载第%s张图片时失败: %s' % (str(i + 1), str(pic_url)))
            print(e)
            continue


if __name__ == '__main__':
    prepare_file_system()
    print '开始从'+url_init+'下载图片'
    all_pic_urls = []
    onePage_urls, next_url = get_onePage_urls(url_init)
    all_pic_urls.extend(onePage_urls)
    #print "初始化时已经添加的页数",len(all_pic_urls)
    page_count = 1  # 累计翻页数
    print('第%s页' % page_count)
    while page_count < 50:
        onePage_urls, next_url = get_onePage_urls(next_url)
        page_count += 1
        print('第%s页' % page_count)
        if next_url == '' or onePage_urls == []:
            break
        all_pic_urls.extend(onePage_urls)

    down_pic(list(set(all_pic_urls)))