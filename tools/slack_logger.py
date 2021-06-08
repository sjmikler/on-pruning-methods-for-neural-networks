import random
import time


class SlackLogger:
    def __init__(self, config, host, desc):
        import slack
        self.host = host
        self.desc = desc
        self.config = config
        self.client = slack.WebClient(config.token)
        self.channel2thread_short = {}
        self.queue_final = []
        self.queue_short = []

    def get_description(self):
        return f'_{self.desc}_'

    def add_exp_report(self, exp):
        message = eval(self.config.say, {'exp': exp}, {'exp': exp})
        message = '`' + message + '`'

        if self.config.get('channel_final'):
            self.queue_final.append(message)

        if self.config.get('channel_short'):
            self.queue_short.append(message)
            self.send_short()

    def add_finish_report(self):
        message = f"Experiment on {self.host} is completed!"
        if self.desc:
            message += f"\n{self.get_description()}"
        self.queue_final.insert(0, message)

    def send_message(self, msg, channel, thread_ts=None, retries=2):
        try:
            response = self.client.chat_postMessage(channel=channel,
                                                    text=msg,
                                                    thread_ts=thread_ts)
            print("SLACK LOGGING SUCCESS!")
            return response
        except Exception as e:
            if retries > 0:
                time.sleep(random.random() * 4 + 1)
                self.send_message(msg, channel, thread_ts, retries=retries - 1)
            else:
                print("SLACK LOGGING FAILED!")
                print(e)

    def get_thread_short(self, channel):
        if channel in self.channel2thread_short:
            return self.channel2thread_short[channel]
        else:
            message = f"Experiment on {self.host} is running!"
            if self.desc:
                message += f"\n{self.get_description()}"

            r = self.send_message(message, channel)
            if r and r['ok']:
                self.channel2thread_short[channel] = r['ts']
                return self.channel2thread_short[channel]

    def send_short(self):
        thread = self.get_thread_short(self.config.channel_short)
        if thread:
            while self.queue_short:
                r = self.send_message(msg=self.queue_short[0],
                                      channel=self.config.channel_short,
                                      thread_ts=thread)
                if r and r['ok']:
                    self.queue_short.pop(0)
                else:
                    break

    def finalize(self):
        if self.queue_final:
            self.add_finish_report()
            final_message = '\n'.join(self.queue_final)
            final_message = final_message.replace('`\n`', '\n')
            final_message = final_message.replace('`', '```')
            self.send_message(final_message, channel=self.config.channel_final)

    def finalize_short(self):
        for channel, thread in self.channel2thread_short.items():
            message = f"Experiment is completed!"
            self.send_message(message, channel, thread)

    def interrupt_short(self):
        for channel, thread in self.channel2thread_short.items():
            message = f"Experiment has been interrupted!"
            self.send_message(message, channel, thread)
