# @brief  Image with support for: CUDA, pyenv, PyTorch, and the 'vitcifar10' Python package.
# @author Luis C. Garcia Peraza Herrera (luiscarlos.gph@gmail.com).
# @date   26 October 2022.

# Image info
FROM nvidia/cuda:11.6.2-cudnn8-runtime-ubuntu20.04
LABEL maintainer="luiscarlos.gph@gmail.com"
LABEL version="0.1"
LABEL description="Image with support for: CUDA, pyenv, and PyTorch."

# Install NVIDIA repository keys
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 134
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

# Install commonly used packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt upgrade -y
RUN DEBIAN_FRONTEND=noninteractive apt install -y apt-utils aptitude sudo vim zsh wget curl git net-tools tmux locales build-essential cmake build-essential zlib1g-dev libbz2-dev libreadline-dev libssl-dev libsqlite3-dev libffi-dev liblzma-dev tk-dev ncurses-dev

# Install locales
RUN locale-gen en_US.UTF-8
ENV LANG en_US.UTF-8
ENV LANGUAGE en_US:en
ENV LC_ALL en_US.UTF-8

# Intall oh-my-zsh so we have a nice terminal
ENV HOME="/root"
RUN wget https://raw.github.com/ohmyzsh/ohmyzsh/master/tools/install.sh -O $HOME/install.sh
RUN chmod +x $HOME/install.sh 
RUN $HOME/install.sh --unattended
RUN rm $HOME/install.sh
RUN sed -i 's/robbyrussell/terminalparty/g' $HOME/.zshrc

# Install Python's pyenv environment
ENV PYTHON_VERSION="3.9.13"
RUN git clone https://github.com/pyenv/pyenv.git $HOME/.pyenv
RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> $HOME/.zshrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> $HOME/.zshrc
RUN echo 'eval "$(pyenv init --path)"' >> $HOME/.zshrc
RUN /bin/zsh -c "source $HOME/.zshrc && pyenv install $PYTHON_VERSION"
RUN /bin/zsh -c "source $HOME/.zshrc && pyenv global $PYTHON_VERSION"

# Install PyTorch environment, more versions here: https://pytorch.org/get-started/previous-versions
ENV PYTORCH_INSTALL_CMD="pip install torch==1.12.1+cu116 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116"
RUN /bin/zsh -c "source $HOME/.zshrc && $PYTORCH_INSTALL_CMD"

# Download and install vitcifar-10 Python module
RUN cd $HOME && git clone https://github.com/luiscarlosgph/vitcifar10.git
RUN /bin/zsh -c "source $HOME/.zshrc && cd $HOME/vitcifar10 && python setup.py install"
RUN rm -r $HOME/vitcifar10

# Start services and run the user-specified command
ADD files/entrypoint.sh /opt/entrypoint.sh
RUN chmod +x /opt/entrypoint.sh
ENTRYPOINT ["/opt/entrypoint.sh"]
CMD ["sleep infinity"]
