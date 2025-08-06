# Start from a stable Debian 12 base
# https://hub.docker.com/r/microsoft/devcontainers-python
FROM freqtradeorg/freqtrade:stable

ARG USERNAME=ftuser
ARG USER_UID=501
ARG USER_GID=20

# Switch to root to modify user and permissions
USER root

# Change UID and update file ownership.
RUN \
    if ! getent group $USER_GID > /dev/null; then groupadd -g $USER_GID $USERNAME; fi && \
    ORIGINAL_UID=$(id -u $USERNAME) && \
    usermod -u $USER_UID -g $USER_GID $USERNAME && \
    find /home/$USERNAME -user $ORIGINAL_UID -exec chown -h $USERNAME:$USER_GID {} + && \
    find /freqtrade -user $ORIGINAL_UID -exec chown -h $USERNAME:$USER_GID {} +

# Switch back to the non-root user
USER $USERNAME

# Used to persist bash history as per https://code.visualstudio.com/remote/advancedcontainers/persist-bash-history
RUN SNIPPET="export PROMPT_COMMAND='history -a' && export HISTFILE=/tmp/commandhistory/.bash_history" \
    && mkdir /tmp/commandhistory \
    && touch /tmp/commandhistory/.bash_history \
    && chown -R $USERNAME /tmp/commandhistory \
    && echo "$SNIPPET" >> "/home/$USERNAME/.bashrc"

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
