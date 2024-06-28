document.getElementById('commandForm').addEventListener('change', updateCommand);

function updateCommand() {
    const folderToAnalyse = document.getElementById('folderToAnalyse').value;
    const descriptorsFolder = document.getElementById('descriptorsFolder').value;

    const options = [];
    document.querySelectorAll('input[name="options"]:checked').forEach(option => {
        options.push(option.value);
    });

    const umap = document.getElementById('umap').value;
    const verbose = document.getElementById('verbose').checked ? 'True' : 'False';

    const command = `python src/preprocess.py ${folderToAnalyse} ${descriptorsFolder} ${options.join(' ')} ${umap} ${verbose}`;
    document.getElementById('commandOutput').value = command;
}

function copyCommand() {
    const commandOutput = document.getElementById('commandOutput');
    commandOutput.select();
    document.execCommand('copy');
    alert('Command copied to clipboard!');
}

// Initial command update
updateCommand();
