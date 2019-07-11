import sys
from hanziconv import HanziConv
import re
import os
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt


def raw_data():
    writer = open('../../Data/zhwiki-categorylinks.txt', 'w')
    i = 1
    with open('../../Data/zhwiki-20181220-categorylinks.sql', 'rb') as reader:
        for line in reader:
            try:
                line = line.decode('utf-8').strip()
                if line.startswith('--') or line.startswith('/*!') or line is '':
                    pass
                else:
                    writer.write(line + '\n')
                # print('line:', i)
            except:
                print('[ERROR]:' + str(i))
            i += 1
    writer.close()


def get_cates():
    # cates = []
    # with open('../../data/categories.txt', 'r', encoding='utf8') as reader:
    #     for line in reader:
    #         info = line.strip().split('\t')
    #         sup = info[0]
    #         if sup == 'PER'
    #             sub = info[1]
    #             if '\\n' in sub:
    #                 sub = sub.split('\\n')[1]
    #             if sub not in cates:
    #                 print(sub)
    #                 cates.append(sub)
    #
    # cates.sort()
    # writer = open('../../Data/per-categories.txt', 'w', encoding='utf8')
    # for cate in cates:
    #     writer.write(cate + '\n')
    # writer.close()

    sups = []
    with open('../../data/per-categories.txt', 'r') as reader:
        for line in reader:
            sups.append(line.strip())

    cates = []
    with open('../../data/categories.txt', 'r') as reader:
        for line in reader:
            info = line.strip().split('\t')
            sup = info[0]
            if sup in sups:
                sub = info[1]
                if '\\n' in sub:
                    sub = sub.split('\\n')[1]
                if sub not in cates:
                    print(sub)
                    cates.append(sub)

    cates.sort()
    writer = open('../../data/per-categories.txt', 'a')
    for cate in cates:
        writer.write(cate + '\n')
    writer.close()


def get_titles():
    cates = []
    with open('raw/per-categories.txt', 'r') as reader:
        for line in reader:
            cates.append(line.strip())

    pages = []
    title = ''
    new_page = False
    with open('raw/zhwiki-20181220-pages-articles-multistream.xml', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if '</page>' in line:
                new_page = False
                title = ''
            if new_page:
                if '<title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                if '[[Category:' in line:
                    cate = line.split('[[Category:')[1].split('| ]]')[0]
                    if cate in cates and title not in pages:
                        pages.append(title)
                        print(title)

    pages.sort()
    with open('raw/per-titles.txt', 'w') as writer:
        for page in pages:
            writer.write(page + '\n')


def get_pages():
    titles = []
    with open('labeled/per-titles.txt', 'r') as reader:
        for line in reader:
            titles.append(line.strip())

    page = ''
    new_page = False
    writer = open('raw/per-pages.txt', 'w')
    with open('raw/zhwiki-20181220-pages-articles-multistream.xml', 'r') as reader:
        for line in reader:
            line = line.rstrip()
            page += (line + '\n')
            if '<title>' in line:
                title = line.split('<title>')[1].split('</title>')[0]
                if title in titles:
                    new_page = True
            if '</page>' in line:
                if new_page:
                    writer.write(page)
                new_page = False
                page = ''
    writer.close()


def t2s():
    count = 0
    writer = open('../../data/wiki/per-pages-s.txt', 'w')
    with open('../../data/wiki/per-pages.txt', 'r') as reader:
        for line in reader:
            writer.write(HanziConv.toSimplified(line))
            count += 1
            print('line:', count)
    writer.close()


def filter():
    # count = 1
    # with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
    #     for line in reader:
    #         if '<ns>' in line:
    #             ns = line.split('<ns>')[1].split('</ns>')[0]
    #             if ns != '0':
    #                 print('line:', count)
    #         count += 1

    title = ''
    new_page = False
    has_text = False
    with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if new_page:
                if '<text' in line:
                    has_text = True
                if '</page>' in line:
                    if not has_text:
                        print(title)
                    new_page = False
                    has_text = False
                    title = ''


def get_ids():
    count = 1
    title = ''
    new_page = False
    ids = []
    with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if new_page:
                if '<title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                if '<revision>' in line:
                    line = reader.readline()
                    id = line.split('<id>')[1].split('</id>')[0]
                    ids.append(id + '\t' + title)
                    title = ''
                    new_page = False
            print('line:', count)
            count += 1
    writer = open('../../data/wiki/per-ids.txt', 'w')
    ids.sort()
    for id in ids:
        writer.write(id + '\n')
    writer.close()


def split():
    page = ''
    title = ''
    id = ''
    infobox = ''
    abstract = ''
    categories = []
    with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
        for line in reader:
            pass


def get_abstracts():
    quots = set()
    count = 1
    title = ''
    id = ''
    abstract = ''
    new_page = False
    is_ab = False
    writer = open('../../data/wiki/per-abstracts.txt', 'w')
    with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if new_page:
                if '</page>' in line:
                    substrs = re.findall("\{\{[^ \n\|\}]*[ \n\|\}]", abstract)
                    for oldstr in substrs:
                        if not str(oldstr).endswith('}'):
                            quots.add(oldstr)

                    abstract = re.sub('      <text xml:space="preserve">', '', abstract)
                    abstract = re.sub('&lt;ref&gt;((?!/ref&gt).)*&lt;/ref&gt;', '', abstract)
                    # abstract = re.sub('&lt;ref((?!/ref&gt).)*&lt;/ref', '', abstract)
                    abstract = re.sub('\{\{[dD]ead link[^\{\}]*\}\}', '', abstract)
                    abstract = re.sub('\{\{[wW]ebarchive[^\{\}]*\}\}', '', abstract)

                    substrs = re.findall("-\{[^\{\}]*\}-", abstract)
                    for oldstr in substrs:
                        newstr = '-[' + oldstr[2:-2] + ']-'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[lL]ang[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[bB][dD][^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[lL]ink[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[tT]sl[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[lL]e[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[tT]ranslink[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    substrs = re.findall("\{\{[iI]nternal[^\}]+\}\}", abstract)
                    for oldstr in substrs:
                        newstr = '[@' + oldstr[2:-2] + '@]'
                        abstract = abstract.replace(oldstr, newstr)

                    abstract = abstract.replace('《{{', '《[@')
                    abstract = abstract.replace('}}》', '@]》')
                    abstract = abstract.replace('{title}', '')

                    while '{{' in abstract:
                        abstract = re.sub('\{\{[^\{\}]*\}\}', '', abstract)
                        # print(abstract+'\n')

                    abstract = re.sub('\{\|[^\}]*\|\}', '', abstract)
                    abstract = abstract.replace('&lt;!-- 学经历信息开始 --&gt;', '')

                    writer.write('************************************\n')
                    writer.write('@id:' + id + '+' + title + '\n\n')
                    writer.write('abstract:' + '\n')
                    writer.write(abstract + '\n\n')
                    new_page = False
                    is_ab = False
                    title = ''
                    id = ''
                    abstract = ''
                if '<title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                if '<revision>' in line:
                    line = reader.readline()
                    id = line.split('<id>')[1].split('</id>')[0]
                if '<text' in line:
                    is_ab = True
                if '</text>' in line or line.startswith('==') or line.startswith('[[Category:'):
                        is_ab = False
                if is_ab:
                    abstract += line
            print('line:', count, '/9798368')
            count += 1

    writer.close()
    writer = open('../../data/wiki/per-quots.txt', 'w')
    quots = list(quots)
    quots.sort()
    for q in quots:
        writer.write(q + '\n')
    writer.close()


def abstracts_process():
    id = ''
    abstract = ''
    new_abstract = False
    writer = open('../../data/wiki/per-abstracts-p.txt', 'w')
    with open('../../data/wiki/per-abstracts.txt', 'r') as reader:
        for line in reader:
            if '************************************' in line:
                abstract.lstrip()
                abstract = re.sub('&lt;[rR][eE][fF]((?!/ref&gt)[\s\S])*&lt;/[rR][eE][fF]&gt;', '', abstract)
                abstract = re.sub('&lt;[rR][eE][fF]((?!&gt)[\s\S])*&gt;', '', abstract)
                abstract = re.sub('&lt;[sS]pan((?!/span&gt)[\s\S])*&lt;/[sS]pan&gt;', '', abstract)
                abstract = re.sub('&lt;!--((?!&gt)[\s\S])*--&gt;', '', abstract)
                abstract = re.sub('&lt;gallery&gt;((?!/gallery&gt;)[\s\S])*&lt;/gallery&gt;', '', abstract)
                abstract = re.sub('&lt;small&gt;((?!/small&gt)[\s\S])*&lt;/small&gt;', '', abstract)
                abstract = re.sub('&lt;big&gt;((?!/big&gt)[\s\S])*&lt;/big&gt;', '', abstract)
                abstract = re.sub('&lt;sub&gt;((?!/sub&gt)[\s\S])*&lt;/sub&gt;', '', abstract)
                abstract = re.sub('&lt;sup&gt;((?!/sup&gt)[\s\S])*&lt;/sup&gt;', '', abstract)
                abstract = re.sub('&lt;blockquote&gt;((?!/blockquote&gt;)[\s\S])*&lt;/blockquote&gt;', '', abstract)
                abstract = re.sub('&lt;math&gt;((?!/math&gt)[\s\S])*&lt;/math&gt;', '', abstract)
                abstract = re.sub('&lt;nowiki&gt;((?!/nowiki&gt)[\s\S])*&lt;/nowiki&gt;', '', abstract)
                abstract = re.sub('&lt;div((?!/div&gt)[\s\S])*&lt;/div&gt;', '', abstract)
                abstract = re.sub('&lt;code((?!/code&gt)[\s\S])*&lt;/code&gt;', '', abstract)
                abstract = re.sub('&lt;font((?!/font&gt)[\s\S])*&lt;/font&gt;', '', abstract)
                abstract = re.sub('&lt;noinclude((?!/noinclude&gt)[\s\S])*&lt;/noinclude&gt;', '', abstract)
                abstract = abstract.replace('&quot', '"')
                abstract = re.sub('&lt;[bB][rR]&gt;', '\n', abstract)
                abstract = re.sub('&lt;[bB][rR]/&gt;', '\n', abstract)
                abstract = re.sub('&lt;[bB][rR] &gt;', '\n', abstract)
                abstract = re.sub('&lt;[bB][rR] /&gt;', '\n', abstract)
                abstract = re.sub('&lt;/[bB][rR]&gt;', '\n', abstract)
                abstract = re.sub('&lt;/[bB][rR] &gt;', '\n', abstract)

                substrs = re.findall("(-\[([a-zA-Z]\|){0,1}zh((?!\]-)[\s\S])*\]-)", abstract)
                for oldstr in substrs:
                    newstr = ''
                    oldstr = str(oldstr[0])
                    if 'zh-hans:' in oldstr:
                        newstr = oldstr.split('zh-hans:')[1].split(';')[0]
                    elif 'zh-cn:' in oldstr:
                        newstr = oldstr.split('zh-cn:')[1].split(';')[0]
                    elif 'zh-hant:' in oldstr:
                        newstr = oldstr.split('zh-hant:')[1].split(';')[0]
                    else:
                        print(oldstr)
                    abstract = abstract.replace(oldstr, newstr)

                writer.write('************************************\n')
                writer.write(id + '\n\n')
                writer.write('abstract:\n' + abstract + '\n\n')
                new_abstract = False
                id = ''
                abstract = ''
            if line.startswith('@id'):
                id = line.strip()
            if line.startswith('abstract:'):
                new_abstract = True
                line = reader.readline()
            if new_abstract:
                if not line.startswith('[[File:') and not line.startswith('[@Link style') and line != '\n':
                    abstract += line

    writer.close()


def get_quots():
    abstract = ''
    new_ab = False
    writer = open('../../data/wiki/per-abstracts-l.txt', 'w')
    with open('../../data/wiki/per-abstracts.txt', 'r') as reader:
        for line in reader:
            if line.startswith('abstract:'):
                writer.write(line)
                line = reader.readline()
                new_ab = True
            if line.startswith('************************************'):
                abstract = abstract.strip()

                substrs = re.findall("(\[\[((?!\]\]).)*\]\])", abstract)
                for oldstr in substrs:
                    oldstr = str(oldstr[0])
                    if '|' in oldstr:
                        newstr = '[[' + oldstr.split('|')[1]
                        abstract = abstract.replace(oldstr, newstr)


                substrs = re.findall("(\[http((?!\]).)*\])", abstract)
                for oldstr in substrs:
                    oldstr = str(oldstr[0])
                    if ' ' in oldstr:
                        tmp = oldstr.split(' ')[0]
                        newstr = oldstr.split(tmp)[1][1:-1]
                        print('tmp:', tmp)
                        print('newstr:', newstr)
                        print('oldstr:', oldstr)
                        abstract = abstract.replace(oldstr, newstr)
                    else:
                        abstract = abstract.replace(oldstr, '')

                substrs = re.findall("(-\[((?!\]).)*\]-)", abstract)
                for oldstr in substrs:
                    oldstr = str(oldstr[0])
                    newstr = oldstr[2:-2]
                    abstract = abstract.replace(oldstr, newstr)

                writer.write(abstract + '\n\n')
                new_ab = False
                abstract = ''
            if new_ab:
                abstract += line.strip()
            else:
                writer.write(line)

    abstract = abstract.strip()

    substrs = re.findall("(\[\[((?!\]\]).)*\]\])", abstract)
    for oldstr in substrs:
        oldstr = str(oldstr[0])
        if '|' in oldstr:
            newstr = oldstr.split('|')[0] + ']]'
            abstract = abstract.replace(oldstr, newstr)

    substrs = re.findall("(\[http((?!\]).)*\])", abstract)
    for quot in substrs:
        oldstr = str(oldstr[0])
        if ' ' in oldstr:
            tmp = oldstr.split(' ')[0]
            newstr = oldstr.split(tmp)[1][1:-1]
            abstract = abstract.replace(oldstr, newstr)
        else:
            abstract = abstract.replace(oldstr, '')

    substrs = re.findall("(-\[((?!\]).)*\]-)", abstract)
    for oldstr in substrs:
        newstr = oldstr[0][2, -2]
        abstract = abstract.replace(oldstr, newstr)

    writer.write(abstract + '\n')
    writer.close()


def get_text():
    ids = []
    writer = open('../../data/wiki/per-text.txt', 'w')
    with open('../../data/wiki/per-abstracts.txt', 'r') as reader:
        for line in reader:
            if line.startswith('abstract:'):
                writer.write(line)
                abstract = reader.readline()

                substrs = re.findall("(\[\[((?!\]\]).)*\]\])", abstract)
                for oldstr in substrs:
                    oldstr = str(oldstr[0])
                    newstr = oldstr[2:-2]
                    abstract = abstract.replace(oldstr, newstr)

                abstract = abstract.replace("'''", '')
                writer.write(abstract)
            else:
                if line.startswith('@id:'):
                    ids.append(line)
                writer.write(line)
    writer.close()

    ids.sort()
    writer = open('../../data/wiki/per-ids.txt', 'w')
    for id in ids:
        writer.write(id)
    writer.close()


def get_infobox():
    cates = []
    infobox = ''
    id = ''
    title = ''
    count = 0
    new_page = False
    new_info = False
    writer_i = open('../../data/wiki/per-infoboxes.txt', 'w')
    writer_c = open('../../data/wiki/per-categories.txt', 'w')
    with open('../../data/wiki/per-pages-s.txt', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if new_page:
                if '<title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                if '<revision>' in line:
                    line = reader.readline()
                    id = line.split('<id>')[1].split('</id>')[0]

                if '{{infobox' in line or '{{Infobox' in line or '{{人物' in line or '{{香港立法局议员' in line\
                        or '{{配音' in line or '{{跆拳道选手' in line or '{{艺术家' in line or '{{艺人' in line\
                        or '{{红楼梦人物' in line or '{{网民信息框' in line or '{{演员资讯框' in line\
                        or '{{声优' in line or '{{军人' in line or '{{作家' in line or '{{学术研究工作者' in line\
                        or '{{明清人物信息框' in line or '{{篮球' in line or '{{相声小品演员' in line:
                    new_info = True
                if new_info:
                    infobox += line
                    count += line.count('{')
                    count -= line.count('}')
                    if count <= 0:
                        new_info = False
                else:
                    if '[[Category:' in line or '[[category:' in line or '[[分类:' in line:
                        cates.append(line)

                if '</page>' in line:
                    writer_i.write('@id:' + id + '+' + title + '\n\n')
                    writer_i.write('infobox:\n' + infobox + '\n')
                    writer_i.write('************************************\n')

                    writer_c.write('@id:' + id + '+' + title + '\n\n')
                    writer_c.write('categories:\n')
                    cates.sort()
                    for c in cates:
                        writer_c.write(c)
                    writer_c.write('\n************************************\n')

                    if count != 0:
                        print('@id:' + id + '+' + title)
                    new_page = False
                    new_info = False
                    cates = []
                    infobox = ''
                    id = ''
                    title = ''
                    count = 0
    writer_i.close()
    writer_c.close()


def infobox_process():
    start = False
    id = ''
    infobox = ''
    categories = []
    writer_i = open('../../data/wiki/per-infoboxes-l.txt', 'w')
    writer_c = open('../../data/wiki/per-categories-l.txt', 'w')
    with open('../../data/wiki/per-infoboxes.txt', 'r') as reader:
        for line in reader:
            if line.startswith('@id:'):
                id = line
            if line.startswith('infobox:'):
                start = True
            if line.startswith('************************************'):
                writer_i.write(id + '\n')
                writer_i.write('infobox:\n' + infobox)
                writer_i.write('\n************************************\n')
                if len(categories) > 0:
                    writer_c.write(id + '\n\ncategories:\n')
                    for c in categories:
                        writer_c.write(c)
                    writer_c.write('\n************************************\n')
                start = False
                id = ''
                infobox = ''
                categories = []
            if start:
                if line.startswith('|') and '=' in line:
                    infobox += line
                if line.startswith('[[Category') or line.startswith('[[分类'):
                    categories.append(line)


def category_process():
    writer = open('../../data/wiki/per-categories-l.txt', 'w')
    id = ''
    category = ''
    start = False
    with open('../../data/wiki/per-categories.txt', 'r') as reader:
        for line in reader:
            if line.startswith('@id'):
                id = line
            if line.startswith('categories:'):
                line = reader.readline()
                start = True
            if line.startswith('************************************'):
                if len(category.strip()) > 0:
                    writer.write(id + '\ncategories:\n' + category)
                    writer.write('\n************************************\n')
                start = False
                id = ''
                category = ''
            if start:
                if '[[' in line:
                    cate = line.split('[[Category:')[1].split(']]')[0].strip()
                    if '|' in cate:
                        cate = cate.split('|')[0].strip()
                    category += (cate + '\n')
    writer.close()


def text_statistic():
    length = set()
    count = 0
    x = [30, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 2000]
    y = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2]
    with open('../../data/wiki/per/per-text.txt', 'r') as reader:
        for line in reader:
            if line.startswith('abstract:'):
                text = reader.readline().strip()
                length.add(len(text))
                if len(text) >= 400:
                    print(text)
                    count += 1

                for i in range(len(x)):
                    if len(text) <= x[i]:
                        y[i] += 1
                        break
    length = list(length)
    length.sort()
    print(length)
    print(count)

    plt.figure()
    plt.plot(x, y)

    # 设置数字标签
    for a, b in zip(x, y):
        plt.text(a, b, b, ha='center', va='bottom', fontsize=10)

    for a, b in zip(x, y):
        plt.text(a, -3000, a, ha='center', va='bottom', fontsize=10)

    plt.title('Number of text in different length')
    plt.xlabel('文本长度')
    plt.ylabel('文本数')
    plt.savefig("../../data/wiki/statistic/文本长度统计.png")
    plt.show()


def attribute_statistic():
    attr_dic = {}
    with open('../../data/wiki/per-infoboxes.txt', 'r') as reader:
        for line in reader:
            if line.startswith('|') and '=' in line:
                attr = line.split('|')[1].split('=')[0].strip()
                attr = attr.lower()
                if attr not in attr_dic:
                    attr_dic[attr] = 0
                if len(line.split('=')) > 1:
                    val = '='.join(line.split('=')[1:]).strip()
                    if len(val) > 0:
                        attr_dic[attr] += 1
    l = sorted(attr_dic.items(), key=lambda x: x[1], reverse=True)
    print(l)
    writer = open('../../data/wiki/attr-val-statistic.txt', 'w')
    for key, value in l:
        writer.write(key + '\t:\t' + str(value) + '\n')
    writer.close()


def temp():
    schema = {}
    with open('../../data/wiki/per/per-schema.txt', 'r') as reader:
        for line in reader:
            kvs = line.strip().split('\t')
            schema[kvs[0]] = kvs[1:]

    writer = open('../../data/wiki/per/per-infoboxes-l.txt', 'w')
    with open('../../data/wiki/per/per-infoboxes.txt', 'r') as reader:
        for line in reader:
            if line.startswith('|') and '=' in line:
                key = line.split('=')[0][1:].strip().lower()
                value = '='.join(line.strip().split('=')[1:])
                for k,v in schema.items():
                    if key in v:
                        writer.write(k + '\t=\t' + value + '\n')
            else:
                writer.write(line)
    writer.close()


def count():
    title = ''
    titles = set()
    new_page = False
    with open('../../data/zhwiki-20181220-pages.xml', 'r') as reader:
        for line in reader:
            if '<page>' in line:
                new_page = True
            if '</page>' in line:
                new_page = False
                title = ''
            if new_page:
                if '<title>' in line:
                    title = line.split('<title>')[1].split('</title>')[0]
                if '[[Category:' in line:
                    if '农作物' in line:
                        titles.add(title)
    print(titles)
    print(len(titles))


def text2instance(path_text, path_instance):
    flist = os.listdir(path_text)
    for file in flist:
        position_word = 1
        position_sentence = 1
        reader = open(path_text+file, 'r')
        writer = open(path_instance+file, 'w')
        for line in reader:
            line = line.split('\t:\t')
            word = line[0].strip()
            ner = line[1].strip()
            label = line[2].strip()
            if len(word) > 0:
                writer.write(word+'\t:\t'+ner+'\t:\t'+str(position_word)+'\t:\t'+str(position_sentence)+'\t:\t'+label+'\n')
                position_word += 1
            if word == '。' or word == '？' or word == '！':
                position_sentence += 1
                position_word = 1
        reader.close()
        writer.close()


def remove_repetition(all_path, repetition_path):
    all_list = os.listdir(all_path)
    repetition_list = os.listdir(repetition_path)
    for file in repetition_list:
        if file in all_list:
            os.remove(all_path+file)


if __name__ == '__main__':
    # get_cates()
    # get_titles()
    # get_pages()
    # t2s()
    # filter()
    # get_ids()
    # get_abstracts()
    # abstracts_process()
    # get_quots()
    # get_text()
    # get_infobox()
    # infobox_process()
    # category_process()
    text_statistic()
    # attribute_statistic()
    # temp()
    # count()
    # text2instance('data/labeled2089_ner/', 'data/labeled/')
    # remove_repetition('data/labeled2089/', 'data/labeled2089_ner')


