using Xunit;

// Язык интерфейса — общий для процесса (Localizer.Current, 3.14): тест смены языка не должен
// менять тексты в параллельно идущих тестах.
[assembly: CollectionBehavior(DisableTestParallelization = true)]
