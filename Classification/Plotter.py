import re
import matplotlib.pyplot as plt

def read_markdown_file(file_path):
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def extract_data_from_markdown(content):
    epochs = []
    train_loss = []
    train_accuracy = []
    val_loss = []
    val_accuracy = []
    test_accuracy = []

    pattern = re.compile(r'Epoch (\d+) \(\d+:\d+\.\d+\)- Training Loss: (\d+\.\d+) - Training Accuracy: (\d+\.\d+)% - Validation Loss: (\d+\.\d+) - Validation Accuracy: (\d+\.\d+)% - Test Accuracy: (\d+)/(\d+) \((\d+\.\d+)%\)')

    matches = pattern.findall(content)

    for match in matches:
        epoch, t_loss, t_accuracy, v_loss, v_accuracy, test_correct, test_total, test_accuracy_percent = match
        epochs.append(int(epoch))
        train_loss.append(float(t_loss))
        train_accuracy.append(float(t_accuracy))
        val_loss.append(float(v_loss))
        val_accuracy.append(float(v_accuracy))
        test_accuracy.append(float(test_accuracy_percent))

    return epochs, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy

def plot_comparison_graphs(epochs, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy):
    plt.figure(figsize=(15, 10))

    # Validation Accuracy vs Training Accuracy
    plt.subplot(2, 2, 1)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, val_accuracy, label='Validation Accuracy')
    plt.title('Training Accuracy vs Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Training Loss vs Validation Loss
    plt.subplot(2, 2, 2)
    plt.plot(epochs, train_loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training Loss vs Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    # Test Accuracy vs Training Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(epochs, train_accuracy, label='Training Accuracy')
    plt.plot(epochs, test_accuracy, label='Test Accuracy')
    plt.title('Training Accuracy vs Test Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    file_path = 'Trained_Networks/Training-18-11-2023/README.md'
    markdown_content = read_markdown_file(file_path)
    epochs, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy = extract_data_from_markdown(markdown_content)
    plot_comparison_graphs(epochs, train_loss, train_accuracy, val_loss, val_accuracy, test_accuracy)

