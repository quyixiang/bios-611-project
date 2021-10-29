FROM rocker/verse 
RUN wget https://deb.nodesource.com/setup_16.x 
RUN bash setup_16.x
RUN apt update && apt-get install -y nodejs
RUN apt update && apt-get install -y emacs openssh-server python3-pip
RUN pip3 install --pre --user hy
RUN pip3 install beautifulsoup4 theano tensorflow keras sklearn pandas numpy pandasql 
RUN ssh-keygen -A
RUN mkdir -p /run/sshd
RUN sudo usermod -aG sudo rstudio
RUN R -e "devtools::install_github('gastonstat/arcdiagram')"
RUN R -e "install.packages(c('matlab','Rtsne'));"
RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y xfce4-terminal gnome-terminal dh-autoreconf libcurl4-gnutls-dev libexpat1-dev gettext libz-dev libssl-dev asciidoc xmlto docbook2x
RUN git clone git://git.kernel.org/pub/scm/git/git.git
WORKDIR /git
RUN make configure &&\
 ./configure --prefix=/usr &&\
 make all doc info &&\
 make install install-doc install-html install-info
WORKDIR /
RUN apt update -y && apt install -y python3-pip
RUN R -e "install.packages(\"reticulate\")"
RUN R -e "install.packages(\"shiny\")"
RUN R -e "install.packages(\"ggplot2\")"
RUN R -e "install.packages(\"tibble\")"
RUN R -e "install.packages(\"tidyr\")"
RUN R -e "install.packages(\"dplyr\")"
RUN R -e "install.packages(\"mgcv\")"
RUN R -e "install.packages(\"gratia\")"
RUN R -e "install.packages(\"patchwork\")"
RUN R -e "remotes::install_github(\"clauswilke/colorblindr\")"
RUN R -e "remotes::install_github(\"clauswilke/relayer\")"
RUN pip3 install numpy pandas sklearn
RUN apt update -y && apt install -y python3-pip
RUN pip3 install jupyter jupyterlab