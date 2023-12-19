# Local-global mcmc with diffusion model

Для сэмплирования из расличных распределений есть целый ряд алгоритмов. Их можно разделить на локальные и глобальные. Однако и у тех, и у других есть недостатки: локальные алгоритмы (ULA, MALA) имеют трудности с мультимодальными распределениями, в то время как глобальные алгоритмы страдают от маленькой вероятности принять новый сэмпл.

В статье Local-Global MCMC kernels: the best of both worlds прелагается чередовать локальные и глобальные шаги.

В данной статье в качестве вспомогательного параметрического распределения используется NormFlows. Мы же хотим использовать Diffusion Model.

![alt text](https://github.com/iarchukdima/mcmc-repo/blob/masterimg/A.png)
![alt text](https://github.com/iarchukdima/mcmc-repo/blob/masterimg/B.png)
![alt text](https://github.com/iarchukdima/mcmc-repo/blob/masterimg/C.png)
