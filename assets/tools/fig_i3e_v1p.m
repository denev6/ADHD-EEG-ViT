data = load('../backup_datasets/ADHD_part1/v1p.mat'); % Load the .mat file
%disp(fieldnames(data)); % {'v1p'}
EEG_signal = data.v1p;
%size(EEG_signal) % 12258, 19

time_vector = 1:12258;

figure;
offset = 3000; % Adjust spacing between channels

hold on;
for ch = 1:19
    plot(time_vector, EEG_signal(:, ch) + ch * offset, 'k');
end
hold off;

title('EEG Channels (v1p)');
yticks(offset * (1:19)); % Set y-axis labels to match channels
yticklabels(arrayfun(@(x) sprintf('Ch %d', x), 1:19, 'UniformOutput', false));
grid on;
