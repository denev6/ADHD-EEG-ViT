data = load('../../backup_datasets/IEEE/ADHD_part1/v1p.mat'); % Load the .mat file
%disp(fieldnames(data)); % {'v1p'}
EEG_signal = data.v1p;
%size(EEG_signal) % 12258, 19

time_vector = 1:12258;

figure;
offset = 3000; % Adjust spacing between channels

hold on;
channel = 1; % random value between 1 and 19
plot(time_vector, EEG_signal(:, channel) + channel * offset, 'k');

title('EEG Channels (v1p-ch1)');
grid on;
