# ~7.3 스터디

Efficient Net 또는 Mobile Net 구현은 본인 branch에서 main branch에 있는 week2_73/efficient_net.py 또는 week2_73/mobile_net.py 파일을 merge하여 작성하시면 됩니다.
이후 파이썬 파일 실행시키면 output과 model summary를 확인할 수 있습니다.

## Merge 하는 방법?
```
git checkout {본인 branch}
```
일단 본인 브랜치로 갑니다.
```
git merge main
```
그다음 main에 있는 파일들을 본인 브랜치로 merge 시킵니다.

merge conflict이 날 경우 수동으로 conflict이 난 파일을 수정합니다. (VS code로 하면 편해요)
또는 지금 컴퓨터에 있는 git 정보를 날린다음 merge합니다.
(본인 브랜치에서)
```
git reset --hard HEAD
git merge main
git add --all
git commit -m "week2 start"
git push
```
