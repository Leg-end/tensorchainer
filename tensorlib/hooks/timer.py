import time
import os
import sys
from collections import OrderedDict
from tensorlib.engine import base_hook


try:
    _get_time = time.perf_counter
except AttributeError:
    if os.name == 'nt':
        _get_time = time.clock
    else:
        _get_time = time.time


class TimerHook(base_hook.Hook):
    """
    Example:
        Code example::
            hook = TimerHook()
            with hook:
                trainer.run()
            hook.print_report()
        Output example::
              LinkName  ElapsedTime  Occurrence
                Linear     41.42sec        2100
                   MLP     42.09sec         700
            Classifier     42.39sec         700
    """
    name = 'TimerHook'
    table = {'sec': 1, 'ms': 10 ** 3, 'us': 10 ** 6, 'ns': 10 ** 9}

    def __init__(self):
        self.call_history = []
        self._running_stack = []
        self._depth = 0
        self._total_time = 0

    def _pre_process(self):
        start = _get_time()
        self._running_stack.append(start)
        self._depth += 1

    def before_forward(self, layer, args, kwargs):
        self._pre_process()

    def _post_process(self, layer):
        start = self._running_stack.pop()
        stop = _get_time()
        elapsed = stop - start
        self.call_history.append((layer.__class__.__name__, elapsed))
        assert self._depth > 0
        self._depth -= 1
        if self._depth == 0:
            self._total_time += elapsed

    def after_forward(self, layer, outputs, args, kwargs):
        self._post_process(layer)

    def total_time(self):
        return self._total_time

    def summary(self):
        summary = OrderedDict()
        for name, elapsed in self.call_history:
            if name not in summary:
                summary[name] = {'elapsed': 0, 'occurrence': 0}
            record = summary[name]
            record['elapsed'] += elapsed
            record['occurrence'] += 1
        return summary

    @staticmethod
    def _choose_unit(second):
        factor = 1
        for unit in ['sec', 'ms', 'us']:
            if second * factor >= 1:
                return factor, unit
            factor *= 1000.0
        return factor, 'ns'

    def print_report(self, unit='auto', file=sys.stdout):
        """
        :param unit: str value in ['sec', 'ms', 'us', 'ns', 'auto(default)', 'autoforeach']
        :param file: output direction
        :return: None
        """
        factor = 1
        entries = [['Layer', 'Elapsed', 'Occurrence']]
        auto_foreach = (unit == 'auto_foreach')
        if unit == 'auto':
            max_time = max(record['elapsed'] for record in self.summary().values())
            factor, unit = self._choose_unit(max_time)
        elif not auto_foreach:
            factor = self.table[unit]
        for name, record in self.summary().items():
            second = record['elapsed']
            if auto_foreach:
                factor, unit = self._choose_unit(second)
            elapsed = '%3.2f%s' % (second * factor, unit)
            occurrence = str(record['occurrence'])
            entries.append([name, elapsed, occurrence])
        entry_widths = [max(len(f) for f, _, _ in entries),
                        max(len(e) for _, e, _ in entries),
                        max(len(o) for _, _, o in entries)]
        template = '  '.join('{:>%d}' % w for w in entry_widths)
        for name, elapsed, occurrence in entries:
            line = template.format(name, elapsed, occurrence)
            file.write(line + '\n')
        file.flush()
